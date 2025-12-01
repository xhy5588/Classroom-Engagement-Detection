import cv2
import mediapipe as mp
import time
import urllib.request
import os
import threading

# Import MediaPipe tasks
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Model configuration
MODEL_FILENAME = 'gesture_recognizer.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'

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

def result_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with lock:
        latest_result = result

def main():
    download_model()

    # Initialize GestureRecognizer
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_FILENAME),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        num_hands=2
    )

    with GestureRecognizer.create_from_options(options) as recognizer:
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
            timestamp_ms = int(time.time() * 1000)
            recognizer.recognize_async(mp_image, timestamp_ms)
            
            # Draw latest results
            with lock:
                current_result = latest_result

            if current_result:
                # Draw landmarks
                if current_result.hand_landmarks:
                    for hand_landmarks in current_result.hand_landmarks:
                        for landmark in hand_landmarks:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # Display detected gestures
                if current_result.gestures:
                    for i, gestures in enumerate(current_result.gestures):
                        # Get the top gesture
                        if gestures:
                            gesture = gestures[0]
                            category_name = gesture.category_name
                            score = gesture.score
                            
                            # Display text on screen
                            # Position text based on hand index (roughly)
                            y_pos = 50 + (i * 30)
                            text = f"Hand {i+1}: {category_name} ({score:.2f})"
                            cv2.putText(frame, text, (10, y_pos), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Show the frame
            cv2.imshow('MediaPipe Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


