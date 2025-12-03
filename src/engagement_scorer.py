import numpy as np
import pickle
import os
from collections import deque

class EngagementScorer:
    def __init__(self, model_path="engagement_model.pkl"):
        # History buffers for smoothing the output score
        self.score_history = deque(maxlen=30)  # Smooth over last ~1 second (at 30fps)
        self.pitch_history = deque(maxlen=15)
        self.feature_buffer = deque(maxlen=30) # Window for temporal features

        # Load the trained ML model
        self.model = None
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Successfully loaded ML model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: Model not found at {model_path}. Using heuristic fallback.")

        # Heuristic weights (Used ONLY if model fails or for specific behavior tagging)
        self.heuristic_weights = {
            'looking_away': -0.3,
            'phone_use': -0.4,
            'yawning': -0.3,
            'sleeping': -0.8
        }
        
        # Store latest class probabilities for visualization
        self.latest_class_probs = {}

    def calculate_score(self, features):
        """
        Calculate engagement score (0.0 to 1.0) based on visual features.
        
        Args:
            features (dict): Dictionary containing:
                - pitch, yaw, roll (Head Pose)
                - ear_left, ear_right, ear (Eye Aspect Ratio)
                - mar (Mouth Aspect Ratio)
        
        Returns: 
            tuple: (smoothed_score, status_text, detected_behaviors)
        """
        if not features:
            return 0.0, "No Face Detected", []

        behaviors = []
        raw_score = 0.5

        # --- 1. ML Model Inference (Primary Method) ---
        # --- 1. ML Model Inference (Primary Method) ---
        if self.model:
            # Add current features to buffer
            self.feature_buffer.append(features)
            
            # Need enough frames to calculate stats (e.g., at least 5)
            if len(self.feature_buffer) < 5:
                return 0.5, "Initializing...", [], {}
                
            # Calculate Window Stats (Mean + Std)
            # Order MUST match training.py: 
            # ['pitch', 'yaw', 'roll', 'ear_left', 'ear_right', 'ear', 'mar', 'gaze_h', 'gaze_v', 'hand_to_mouth', 'hand_to_ear', 'hand_height', 'brow_eye_dist', 'mouth_curvature']
            # for each: mean, std
            
            cols = ['pitch', 'yaw', 'roll', 'ear_left', 'ear_right', 'ear', 'mar', 'gaze_h', 'gaze_v',
                    'hand_to_mouth', 'hand_to_ear', 'hand_height', 'brow_eye_dist', 'mouth_curvature']
            input_vector = []
            
            # Convert buffer to list of dicts for easier processing
            buffer_list = list(self.feature_buffer)
            
            for col in cols:
                values = [f.get(col, 0) for f in buffer_list]
                mean_val = np.mean(values)
                std_val = np.std(values)
                input_vector.extend([mean_val, std_val])
            
            input_features = [input_vector]
            
            # Predict Class (0 or 1) and Probability
            # model.classes_ is usually [0, 1]. index 1 is 'Engaged'
            try:
                probs = self.model.predict_proba(input_features)[0]
                classes = self.model.classes_

                # Map Activity to Engagement Score
                # Define mapping here or load from shared config
                ENGAGED_CLASSES = {"neutral", "frowning", "raisehand", "neutralface"}
                
                # Calculate raw score by summing probabilities of engaged classes
                raw_score = 0.0
                for cls, p in zip(classes, probs):
                    if str(cls) in ENGAGED_CLASSES:
                        raw_score += p
                
                # Capture all class probabilities
                if hasattr(self.model, 'classes_'):
                    self.latest_class_probs = {
                        str(cls): p for cls, p in zip(self.model.classes_, probs)
                    }
                
            except Exception as e:
                print(f"Prediction Error: {e}")
                raw_score = 0.0 # Fallback
        else:
            print("Error: No model loaded. Cannot calculate score.")
            return 0.0, "Model Error", [], {}
        # if len(self.pitch_history) >= 10:
        #     pitch_var = np.var(self.pitch_history)
        #     if pitch_var > 15: # High variance = movement
        #         raw_score = max(raw_score, 0.85) # Force High Score
        #         behaviors.append("Nodding")

        # # --- 2. Heuristic Fallback (Secondary Method) ---
        # else:
        #     # Simple rule-based scoring if model is missing
        #     raw_score = 0.8 # Start high

        #     # Penalize for looking away
        #     if abs(features.get('yaw', 0)) > 30:
        #         raw_score += self.heuristic_weights['looking_away']
        #         behaviors.append("Looking Away")
                
        #     # Penalize for looking down (phone)
        #     if features.get('pitch', 0) < -20 or features.get('gaze_v', 0.0) > 0.1:
        #         raw_score += self.heuristic_weights['phone_use']
        #         if features.get('gaze_v', 0.0) > 0.1 and features.get('pitch', 0) < -10:
        #             behaviors.append("Looking at Watch/Phone")
        #         elif features.get('pitch', 0) < -20:
        #             behaviors.append("Head Down")

        #     raw_score = max(0.0, min(1.0, raw_score))

        # # --- 3. Specific Behavior Detection (For User Feedback) ---
        # # Even if the ML gives the score, we want to tell the user WHY.
        
        # # Check Yaw (Looking away)
        # if abs(features.get('yaw', 0)) > 30:
        #     direction = "Left" if features.get('yaw', 0) > 0 else "Right"
        #     behaviors.append(f"Looking {direction}")

        # # Check Pitch (Looking down/Phone)
        # if features.get('pitch', 0) < -25:
        #     behaviors.append("Looking Down")
        # elif features.get('pitch', 0) > 25:
        #     behaviors.append("Looking Up")

        # # Check Eyes (Sleeping)
        # if features.get('ear', 0) < 0.15:
        #     behaviors.append("Eyes Closed")

        # # Check Mouth (Yawning)
        # if features.get('mar', 0) > 0.5:
        #     behaviors.append("Yawning")
        #     raw_score -= 0.4
        
        # --- 4. Smoothing ---
        # Use Exponential Moving Average to prevent jitter
        if self.score_history:
            prev_score = self.score_history[-1]
            smoothed_score = (raw_score * 0.15) + (prev_score * 0.85)
        else:
            smoothed_score = raw_score
            
        self.score_history.append(smoothed_score)

        # --- 5. Strict Binary Status ---
        # STRICT THRESHOLD: 0.5
        if smoothed_score >= 0.5:
            status = "Engaged"
        else:
            status = "Not Engaged"

        # --- 6. Class Probabilities ---
        # Return the captured probabilities
        return smoothed_score, status, behaviors, self.latest_class_probs

