import numpy as np
import pickle
import os
from collections import deque

class EngagementScorer:
    def __init__(self, model_path="engagement_model.pkl"):
        # History buffers for smoothing the output score
        self.score_history = deque(maxlen=30)  # Smooth over last ~1 second (at 30fps)
        self.pitch_history = deque(maxlen=15)

        # Load the trained ML model
        self.model = None
        self.id_to_name = {}
        self.binary_map = {}

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Handle new dictionary format or fallback to old model
                    if isinstance(data, dict) and "model" in data:
                        self.model = data["model"]
                        self.id_to_name = data["id_to_name"]
                        self.binary_map = data["binary_map"]
                    else:
                        self.model = data # Fallback for old binary-only models
                        print("Warning: Old model format detected. Category detection disabled.")
                        
                print(f"Successfully loaded ML model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")


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
        is_engaged_binary = 0

        # --- 1. ML Model Inference (Primary Method) ---
        if self.model:
            # Prepare feature vector in the EXACT order used during training
            # Columns: pitch, yaw, roll, ear_left, ear_right, ear_avg, mar
            input_features = [[
                features.get('pitch', 0),
                features.get('yaw', 0),
                features.get('roll', 0),
                features.get('ear_left', 0),
                features.get('ear_right', 0),
                features.get('ear', 0),      # This corresponds to 'ear_avg' in training
                features.get('mar', 0),
                features.get('gaze_h', 0.5), 
                features.get('gaze_v', 0.0)  
            ]]
            
            # Predict Class (0 or 1) and Probability
            # model.classes_ is usually [0, 1]. index 1 is 'Engaged'
            try:
                # 1. Predict the specific Category ID (0-7)
                pred_id = self.model.predict(input_features)[0]
                # 2. Get the Category Name (e.g., "phone")
                current_category = self.id_to_name.get(pred_id, "Unknown")
                # 3. Get Binary Status (1 = Engaged, 0 = Not)
                is_engaged_binary = self.binary_map.get(pred_id, 0)
                probs = self.model.predict_proba(input_features)[0]
                confidence = probs[pred_id]
                if is_engaged_binary == 1:
                    raw_score = 0.5 + (0.5 * confidence)
                else:
                    raw_score = 0.5 - (0.5 * confidence)

            except Exception as e:
                print(f"Prediction Error: {e}")
                raw_score = 0.0 # Fallback
        else:
            print("Error: No model loaded. Cannot calculate score.")
            return 0.0, "Model Error", []
        
        #Update history
        self.pitch_history.append(features.get('pitch', 0))

        #Check for Nodding (High Variance in Pitch)
        if len(self.pitch_history) >= 10:
            pitch_var = np.var(self.pitch_history)
            if pitch_var > 15:  # Threshold for nodding intensity
                # Force the category to nodding and score to high
                current_category = "nodding"
                raw_score = max(raw_score, 0.85) 
                if "Nodding" not in behaviors:
                    behaviors.append("Nodding")

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

        return smoothed_score, status, current_category

