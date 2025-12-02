import numpy as np
import pickle
import os
from collections import deque
import pandas as pd

class FeatureBuffer:
    def __init__(self, maxlen=30):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, features):
        self.buffer.append(features)
        
    def get_stats(self):
        if not self.buffer:
            return {}
            
        # Convert buffer to DataFrame for easy stats
        df = pd.DataFrame(list(self.buffer))
        
        stats = {}
        # Calculate Mean and Std for key features
        for col in ['pitch', 'yaw', 'roll', 'ear', 'mar', 'hand_face_dist']:
            if col in df.columns:
                stats[f'{col}_mean'] = df[col].mean()
                stats[f'{col}_std'] = df[col].std() if len(df) > 1 else 0.0
                
        return stats

class EngagementScorer:
    def __init__(self, model_path="engagement_model.pkl"):
        # History buffers for smoothing the output score
        self.score_history = deque(maxlen=30)  # Smooth over last ~1 second (at 30fps)
        
        # Feature Buffer for Temporal Features
        self.feature_buffer = FeatureBuffer(maxlen=30)

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

    def calculate_score(self, features):
        """
        Calculate engagement score (0.0 to 1.0) based on visual features.
        
        Args:
            features (dict): Dictionary containing:
                - pitch, yaw, roll (Head Pose)
                - ear_left, ear_right, ear (Eye Aspect Ratio)
                - mar (Mouth Aspect Ratio)
                - hand_face_dist (Hand to Face Distance)
        
        Returns: 
            tuple: (smoothed_score, status_text, detected_behaviors)
        """
        if not features:
            return 0.0, "No Face Detected", []

        # --- Smoothing Raw Features (Anti-Jitter) ---
        # Apply EMA to raw features BEFORE adding to buffer
        # This reduces webcam noise which causes high std dev (false nodding)
        alpha = 0.3 # Lower = more smoothing
        
        if not hasattr(self, 'smoothed_features'):
            self.smoothed_features = features.copy()
        else:
            for key in ['pitch', 'yaw', 'roll', 'ear', 'mar', 'hand_face_dist']:
                if key in features:
                    self.smoothed_features[key] = (features[key] * alpha) + (self.smoothed_features.get(key, 0) * (1 - alpha))
        
        # Add SMOOTHED features to buffer
        self.feature_buffer.add(self.smoothed_features)
        
        behaviors = []
        raw_score = 0.5

        # --- 1. ML Model Inference (Primary Method) ---
        if self.model:
            # Get temporal stats
            stats = self.feature_buffer.get_stats()
            
            # Prepare feature vector (DataFrame)
            input_features = pd.DataFrame([stats])
            
            # Predict Class (Activity) and Probability
            try:
                # Get probabilities for all classes
                probs = self.model.predict_proba(input_features)[0]
                classes = self.model.classes_
                
                # Find max probability and corresponding class
                max_prob_idx = np.argmax(probs)
                predicted_activity = classes[max_prob_idx]
                confidence = probs[max_prob_idx]
                
                # Map Activity to Engagement Score
                # Define mapping here or load from shared config
                ACTIVITY_TO_ENGAGEMENT = {
                    "neutralface": 1.0,
                    "frowning": 1.0,
                    "nodding": 1.0,
                    "drinking": 0.0,
                    "phone": 0.0,
                    "yawning": 0.0,
                    "tilt": 0.0,
                    "raisehand": 1.0,
                    "watch": 0.0
                }
                
                # Default to 0.5 if unknown activity
                raw_score = ACTIVITY_TO_ENGAGEMENT.get(predicted_activity, 0.5)
                
                # Only show specific activity if confidence is high
                CONFIDENCE_THRESHOLD = 0.7
                if confidence > CONFIDENCE_THRESHOLD:
                    behaviors.append(f"{predicted_activity} ({confidence:.2f})")
                
            except Exception as e:
                print(f"Prediction Error: {e}")
                raw_score = 0.5 # Fallback
        
        # --- 2. Heuristic Fallback (Secondary Method) ---
        else:
            # Simple rule-based scoring if model is missing
            raw_score = 0.8 # Start high
            
            # Penalize for looking away
            if abs(features.get('yaw', 0)) > 30:
                raw_score += self.heuristic_weights['looking_away']
            
            # Penalize for looking down (phone)
            if features.get('pitch', 0) < -20:
                raw_score += self.heuristic_weights['phone_use']
                
            raw_score = max(0.0, min(1.0, raw_score))

        # --- 4. Smoothing ---
        # Use Exponential Moving Average to prevent jitter
        if self.score_history:
            prev_score = self.score_history[-1]
            smoothed_score = (raw_score * 0.15) + (prev_score * 0.85)
        else:
            smoothed_score = raw_score

        # --- 3. Specific Behavior Detection (For User Feedback) ---
        # Even if the ML gives the score, we want to tell the user WHY.
        # Only add heuristic behaviors if ML didn't already find something high confidence
        if not behaviors:
            # Check Yaw (Looking away)
            if abs(features.get('yaw', 0)) > 30:
                direction = "Left" if features.get('yaw', 0) > 0 else "Right"
                behaviors.append(f"Looking {direction}")

            # Check Pitch (Looking down/Phone)
            if features.get('pitch', 0) < -25:
                behaviors.append("Looking Down")
            elif features.get('pitch', 0) > 25:
                behaviors.append("Looking Up")

            # Check Eyes (Sleeping)
            if features.get('ear', 0) < 0.15:
                behaviors.append("Eyes Closed")

            # Check Mouth (Yawning)
            if features.get('mar', 0) > 0.5:
                behaviors.append("Yawning")

            # Check Hand Raise
            if features.get('hand_raised', False):
                behaviors.append("Hand Raised")
                # Force high engagement score if hand is raised
                smoothed_score = max(smoothed_score, 0.9)

            # Check Frowning
            # Heuristic: < 0.30 normalized distance often indicates frowning
            if features.get('frown_score', 1.0) < 0.30:
                behaviors.append("Frowning")
                # Frowning might indicate confusion (engaged but struggling) or focus
                # Let's keep it as high engagement but note the behavior
                smoothed_score = max(smoothed_score, 0.8)

        self.score_history.append(smoothed_score)

        # --- 5. Strict Binary Status ---
        # STRICT THRESHOLD: 0.5
        if smoothed_score >= 0.5:
            status = "Engaged"
        else:
            status = "Not Engaged"

        return smoothed_score, status, behaviors

