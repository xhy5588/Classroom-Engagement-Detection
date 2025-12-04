import cv2
import mediapipe as mp
import numpy as np
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
                    score, status, behaviors, prob_dict = scorer.calculate_score(features)
                    
                    # --- UI Display ---
                    
                    # Determine Color based on Strict Binary Status
                    if status == "Engaged":
                        color = (0, 255, 0)   # Green
                        feedback_msg = "Your students are listening. Good job! Keep going..."
                    else:
                        color = (0, 0, 255)   # Red ("Not Engaged")
                        feedback_msg = "Your students may be distracted. Try ask a question or tell a joke."

                    
                    # Line 1: Binary Status
                    cv2.putText(image, f"Status: {status} ({score:.2f})", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Line 2: Detected Category
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

                    # --- Side Panel for Probabilities ---
                    panel_width = 300
                    panel_height = image.shape[0]
                    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
                    
                    if prob_dict:
                        y_offset = 40
                        bar_height = 20
                        
                        # Find max prob for highlighting
                        # prob_dict is {name: prob}
                        max_prob_name = max(prob_dict, key=prob_dict.get)
                        
                        cv2.putText(panel, "Class Probabilities", (10, 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Sort by Name for consistent order
                        sorted_names = sorted(prob_dict.keys())
                        
                        for name in sorted_names:
                            prob = prob_dict[name]
                            
                            # Highlight max probability
                            if name == max_prob_name:
                                text_color = (0, 255, 0) # Green
                                bar_color = (0, 255, 0)
                            else:
                                text_color = (200, 200, 200) # Gray
                                bar_color = (100, 100, 100)
                            
                            # Draw Label
                            cv2.putText(panel, f"{name}: {prob:.2f}", (10, y_offset + 15),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                            
                            # Draw Bar
                            bar_width = int(prob * (panel_width - 20))
                            cv2.rectangle(panel, (10, y_offset + 20), (10 + bar_width, y_offset + 20 + bar_height), bar_color, -1)
                            
                            y_offset += 50

                    # Combine Image and Panel
                    image = cv2.hconcat([image, panel])

            cv2.imshow('Engagement Detection Demo v0', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

