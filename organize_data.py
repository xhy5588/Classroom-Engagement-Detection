import os
import shutil

def organize_data():
    source_dir = "data/clips"
    target_base = "data"
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    # Categories based on training.py
    categories = [
        "neutralface", "frowning", "drinking", "phone", 
        "yawning", "tilt", "raisehand", "watch", "nodding"
    ]
    
    # Create category directories
    for cat in categories:
        os.makedirs(os.path.join(target_base, cat), exist_ok=True)
        
    for filename in os.listdir(source_dir):
        if not filename.endswith(".mp4"):
            continue
            
        # Filename format: {action_label}_{lighting_label}_light_{number}.mp4
        # We need to extract the action_label
        parts = filename.split('_')
        if len(parts) < 2:
            print(f"Skipping malformed filename: {filename}")
            continue
            
        action_label = parts[0]
        
        if action_label in categories:
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_base, action_label, filename)
            shutil.move(source_path, target_path)
            print(f"Moved {filename} to {action_label}/")
        else:
            print(f"Unknown label {action_label} for {filename}")

if __name__ == "__main__":
    organize_data()
