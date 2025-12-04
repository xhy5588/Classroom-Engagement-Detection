import cv2
import mediapipe as mp
from src.visual_feature_extractor import VisualFeatureExtractor
from src.engagement_scorer import EngagementScorer

def main():
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize Modules
    extractor = VisualFeatureExtractor()
    scorer = EngagementScorer()

    cap = cv2.VideoCapture(0)
    
    print("Starting Engagement System. Press 'q' to exit.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 1. Draw Visuals
            if results.pose_landmarks:
                 mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # 2. Extract Features
                features = extractor.extract_features(results.face_landmarks, results.pose_landmarks, image.shape)
                
                # 3. Calculate Engagement
                if features:
                    score, status, behaviors = scorer.calculate_score(features)
                    
                    # --- UI Display ---
                    
                    # Determine Color based on Strict Binary Status
                    if status == "Engaged":
                        color = (0, 255, 0)   # Green
                        feedback_msg = "Your students are listening. Good job! Keep going..."
                    else:
                        color = (0, 0, 255)   # Red ("Not Engaged")
                        feedback_msg = "Your students may be distracted. Try ask a question or tell a joke."

                    
                    # cv2.rectangle(image, (0, 0), (640, 50), (0, 0, 0), -1)
                    # cv2.putText(image, f"Status: {status} ({score:.2f})", (10, 35),
                    #           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # Line 1: Binary Status
                    cv2.putText(image, f"Status: {status} ({score:.2f})", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Line 2: Detected Category
                    # Displaying the specific action (e.g., "Category: Yawning")
                    cv2.putText(image, f"Action: {behaviors}", (10, 65),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    # Line 3: Feedback Message
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9
                    thickness = 2
                    text_size = cv2.getTextSize(feedback_msg, font, font_scale, thickness)[0]
                    
                    # Draw white background rectangle
                    padding = 10
                    cv2.rectangle(image, 
                                (5, 95), 
                                (text_size[0] + padding * 2, 130), 
                                (255, 255, 255), -1)
                    
                    # Draw black text on white background
                    cv2.putText(image, feedback_msg, (10, 120),
                              font, font_scale, (0, 0, 0), thickness)  
                                  
                    # Debug Stats (Bottom)
                    stats = f"Yaw:{features['yaw']:.0f} Pitch:{features['pitch']:.0f} EAR:{features['ear']:.2f}"
                    cv2.putText(image, stats, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow('Engagement Detection Demo v0', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

