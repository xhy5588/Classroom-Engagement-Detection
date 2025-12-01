import cv2
import mediapipe as mp
import time
import urllib.request
import os
import threading

# Import MediaPipe tasks
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Model configuration
MODEL_FILENAME = 'face_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

def download_model():
    if not os.path.exists(MODEL_FILENAME):
        print(f"Downloading {MODEL_FILENAME} from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        print("Download complete.")
    else:
        print(f"{MODEL_FILENAME} already exists.")

# Global variable to store results
latest_result = None
lock = threading.Lock()

def result_callback(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with lock:
        latest_result = result

def main():
    download_model()

    # Initialize FaceLandmarker
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_FILENAME),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        # Initialize OpenCV
        cap = cv2.VideoCapture(0)
        
        print("Starting camera. Press 'q' to exit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process the frame
            # Timestamp must be monotonically increasing
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)
            
            # Draw latest results
            with lock:
                current_result = latest_result

            if current_result:
                 for face_landmarks in current_result.face_landmarks:
                     for landmark in face_landmarks:
                         x = int(landmark.x * frame.shape[1])
                         y = int(landmark.y * frame.shape[0])
                         cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Show the frame
            cv2.imshow('MediaPipe Face Landmarker', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
