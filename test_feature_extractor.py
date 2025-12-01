import cv2
import mediapipe as mp
from src.visual_feature_extractor import VisualFeatureExtractor

def main():
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize Feature Extractor
    extractor = VisualFeatureExtractor()

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    print("Starting Visual Feature Test. Press 'q' to exit.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw Pose Landmarks (for hand raise debug)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            if results.face_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Extract Features (Pass pose_landmarks now)
                features = extractor.extract_features(results.face_landmarks, results.pose_landmarks, image.shape)
                
                if features:
                    # Display Features
                    y_pos = 30
                    for key, value in features.items():
                        if isinstance(value, float):
                            text = f"{key}: {value:.2f}"
                        else:
                            text = f"{key}: {value}"
                            
                        cv2.putText(image, text, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_pos += 25

            cv2.imshow('Visual Features Test', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

