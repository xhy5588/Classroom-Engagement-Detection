import csv
import subprocess
import os
import collections
import argparse

def process_videos(csv_path, output_dir):
    """
    Reads a CSV file and clips videos using ffmpeg.
    
    Args:
        csv_path: Path to the CSV file containing video clip information.
        output_dir: Directory to save the clipped videos.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Counter to track the number of clips for each action_label + lighting_label pair
    clip_counts = collections.defaultdict(int)

    try:
        with open(csv_path, 'r') as csvfile:
            # Check if the file has a header
            sample = csvfile.read(1024)
            csvfile.seek(0)
            has_header = csv.Sniffer().has_header(sample)
            
            if has_header:
                reader = csv.DictReader(csvfile)
                # Verify headers if they exist
                required_headers = ['filename', 'clip_start', 'clip_end', 'action_label', 'lighting_label']
                if reader.fieldnames:
                     missing_headers = [h for h in required_headers if h not in reader.fieldnames]
                     if missing_headers:
                         # Fallback to assuming no header if headers don't match expected
                         # This might happen if the first row looks like a header but isn't the expected one
                         # But Sniffer is heuristic. 
                         # Given the user data, let's just try to read as list if DictReader fails or just assume no header if we know it's the specific file.
                         # But to be robust:
                         pass
            else:
                reader = csv.reader(csvfile)

            for row in reader:
                if isinstance(reader, csv.DictReader):
                    filename = row['filename']
                    clip_start = row['clip_start']
                    clip_end = row['clip_end']
                    action_label = row['action_label']
                    lighting_label = row['lighting_label']
                else:
                    # Assume column order: filename, clip_start, clip_end, action_label, lighting_label
                    if len(row) < 5:
                        continue
                    filename = row[0]
                    clip_start = row[1]
                    clip_end = row[2]
                    action_label = row[3]
                    lighting_label = row[4]

                # Clean timestamps (replace comma with dot for ffmpeg if needed, though ffmpeg often handles comma)
                clip_start = clip_start.replace(',', '.')
                clip_end = clip_end.replace(',', '.')

                # Construct the input video path
                # Assuming the script is run from the project root, and videos are in data/raw_videos/
                video_dir = os.path.join("data", "raw_videos")
                input_video_path = os.path.join(video_dir, filename)

                if not os.path.exists(input_video_path):
                    # Try appending .mp4
                    if os.path.exists(input_video_path + ".mp4"):
                        input_video_path += ".mp4"
                    elif os.path.exists(input_video_path + ".MP4"):
                        input_video_path += ".MP4"
                    else:
                        # Try case-insensitive search in the directory
                        found = False
                        if os.path.exists(video_dir):
                            for f in os.listdir(video_dir):
                                if f.lower() == filename.lower():
                                    input_video_path = os.path.join(video_dir, f)
                                    found = True
                                    break
                                if f.lower() == (filename + ".mp4").lower():
                                    input_video_path = os.path.join(video_dir, f)
                                    found = True
                                    break
                        
                        if not found:
                            print(f"Warning: Input video not found: {filename} (checked {input_video_path} and variants). Skipping.")
                            continue

                # Generate the output filename
                # Format: {action_label}_{lighting_label}_light_{number}
                # number is 5 digits, 0-indexed
                
                key = f"{action_label}_{lighting_label}"
                count = clip_counts[key]
                output_filename = f"{action_label}_{lighting_label}light_{count:05d}.mp4"
                output_path = os.path.join(output_dir, output_filename)
                
                # Increment the counter for the next clip of this type
                clip_counts[key] += 1

                print(f"Processing: {filename} ({clip_start} - {clip_end}) -> {output_filename}")

                # Construct the ffmpeg command
                # Removed -c copy to ensure playback compatibility (re-encoding)
                command = [
                    "ffmpeg",
                    "-y",
                    "-i", input_video_path,
                    "-ss", clip_start,
                    "-to", clip_end,
                    # "-c", "copy", # Removed to fix playback issues
                    output_path
                ]
                
                # Run ffmpeg
                try:
                    # Capture output to avoid cluttering the console, unless error
                    result = subprocess.run(command, capture_output=True, text=True)
                    if result.returncode != 0:
                         print(f"Error processing {filename}: {result.stderr}")
                except FileNotFoundError:
                    print("Error: ffmpeg not found. Please install ffmpeg.")
                    return

    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip videos based on a CSV file.")
    parser.add_argument("--csv", default="data/labeling_support/nodding_labels.csv", help="Path to the CSV file.")
    parser.add_argument("--output", default="data/clips", help="Directory to save output clips.")
    args = parser.parse_args()

    process_videos(args.csv, args.output)
