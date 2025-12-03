import cv2
import numpy as np
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
                    score, status, behaviors, class_probs = scorer.calculate_score(features)
                    
                    # --- UI Display ---
                    
            # Create Canvas (Camera + Sidebar)
            h, w, c = image.shape
            sidebar_w = 300
            canvas = np.zeros((h, w + sidebar_w, c), dtype=np.uint8)
            
            # Copy Camera Image to Left Side
            canvas[0:h, 0:w] = image
            
            # 1. Draw Visuals (on Canvas)
            # Note: Landmarks are drawn on 'image' which is now part of 'canvas'. 
            # If we want landmarks, we should have drawn them on 'image' BEFORE copying.
            # Wait, the code above draws landmarks on 'image' at lines 38 and 43.
            # So 'image' already has landmarks. Correct.
            
            # Re-copy image to canvas to ensure landmarks are visible
            canvas[0:h, 0:w] = image

            # 2. Extract Features (Already done using 'image')
            # features = extractor.extract_features(...) # Done above

            # 3. Calculate Engagement
            if features:
                score, status, behaviors, class_probs = scorer.calculate_score(features)
                
                # --- UI Display (On Canvas) ---
                
                # Status Bar (Top of Camera)
                if status == "Engaged":
                    color = (0, 255, 0)   # Green
                else:
                    color = (0, 0, 255)   # Red
                
                cv2.rectangle(canvas, (0, 0), (w, 50), (0, 0, 0), -1)
                cv2.putText(canvas, f"Status: {status} ({score:.2f})", (10, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                          
                # Behavior List (Overlay on Camera)
                y_pos = 80
                for behavior in behaviors:
                    cv2.putText(canvas, f"• {behavior}", (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    y_pos += 30
                    
                # Probability Distribution (Right Sidebar)
                if class_probs:
                    x_start = w + 20
                    y_start = 50
                    bar_height = 20
                    max_bar_width = sidebar_w - 40
                    
                    cv2.putText(canvas, "Action Probabilities:", (x_start, y_start - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                              
                    # Sort by probability descending
                    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
                    
                    for cls_name, prob in sorted_probs:
                        # Intensity Color
                        intensity = int(prob * 255)
                        bar_color = (0, intensity, 0)
                        
                        # Highlight classified (top one)
                        if prob == sorted_probs[0][1]:
                            text_color = (0, 255, 0) # Bright Green Text
                        else:
                            text_color = (200, 200, 200)
                        
                        width = int(prob * max_bar_width)
                        
                        # Draw Bar
                        cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + bar_height), bar_color, -1)
                        cv2.rectangle(canvas, (x_start, y_start), (x_start + max_bar_width, y_start + bar_height), (50, 50, 50), 1)
                        
                        # Text
                        text = f"{cls_name}: {prob:.2f}"
                        cv2.putText(canvas, text, (x_start, y_start + 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                        
                        y_start += 35

                # Debug Stats (Bottom of Camera)
                stats = f"Yaw:{features['yaw']:.0f} Pitch:{features['pitch']:.0f} EAR:{features['ear']:.2f}"
                cv2.putText(canvas, stats, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow('Engagement Detection Demo v0', canvas)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

