import numpy as np
import pickle
import os
from collections import deque

class EngagementScorer:
    def __init__(self, model_path="engagement_model.pkl"):
        # History buffers for smoothing the output score
        self.score_history = deque(maxlen=30)  # Smooth over last ~1 second (at 30fps)
        self.pitch_history = deque(maxlen=30)
        self.feature_buffer = deque(maxlen=30) # Buffer for temporal features
        self.prob_history = deque(maxlen=10)   # Buffer for probability smoothing

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
            tuple: (smoothed_score, status_text, detected_behaviors, probs, id_to_name)
        """
        if not features:
            return 0.0, "No Face Detected", [], None, self.id_to_name

        behaviors = []
        raw_score = 0.5
        is_engaged_binary = 0
        prob_dict = {}

        
        # Update feature buffer
        self.feature_buffer.append(features)

        # --- 1. Heuristic Checks ---
        # Priority: Drinking > Phone > Raise Hand > Nodding > Model
        
        hand_near_mouth = features.get('hand_near_mouth', 0.0) > 0.5
        roll = features.get('roll', 0.0)
        pitch = features.get('pitch', 0.0)
        mar = features.get('mar', 0.0)
        
        # Refined Drinking: Hand near mouth + Upright Head (not resting) + Mouth closed (not yawning)
        is_drinking = hand_near_mouth and abs(roll) < 15 and mar < 0.5
        
        is_hand_raised = features.get('hand_raised', 0.0) > 0.5
        
        # Phone Heuristic: Looking down significantly
        # Pitch > 20 usually means looking down (depending on coord system, assuming standard)
        is_phone = pitch > 15.0 
        
        is_nodding = False
        if len(self.pitch_history) >= 15:
            pitch_data = np.array(self.pitch_history)
            pitch_std = np.std(pitch_data)
            centered = pitch_data - np.mean(pitch_data)
            signs = np.sign(centered)
            zero_crossings = np.where(np.diff(signs))[0]
            num_crossings = len(zero_crossings)

            # Desensitized Nodding: Higher threshold (3.5) to avoid sudden movements
            if pitch_std > 3.5 and num_crossings >= 2:
                is_nodding = True

        # --- Decision Logic ---
        override_category = None
        
        if is_drinking:
            override_category = "drinking"
            if "Drinking" not in behaviors: behaviors.append("Drinking")
        elif is_phone:
            override_category = "phone"
            if "Phone" not in behaviors: behaviors.append("Phone")
        elif is_hand_raised:
            override_category = "raisehand"
            if "Hand Raised" not in behaviors: behaviors.append("Hand Raised")
        elif is_nodding:
            override_category = "nodding"
            if "Nodding" not in behaviors: behaviors.append("Nodding")

        # --- 2. ML Model Inference ---
        if self.model:
            # ... (Feature prep code) ...
            # Need enough history for temporal features
            if len(self.feature_buffer) < 2: 
                 pitch_mean = features.get('pitch', 0)
                 pitch_std = 0
                 yaw_mean = features.get('yaw', 0)
                 yaw_std = 0
                 roll_mean = features.get('roll', 0)
                 roll_std = 0
                 ear_mean = features.get('ear', 0)
                 ear_std = 0
                 mar_mean = features.get('mar', 0)
                 mar_std = 0
            else:
                 pitch_vals = [f['pitch'] for f in self.feature_buffer]
                 yaw_vals = [f['yaw'] for f in self.feature_buffer]
                 roll_vals = [f['roll'] for f in self.feature_buffer]
                 ear_vals = [f['ear'] for f in self.feature_buffer]
                 mar_vals = [f['mar'] for f in self.feature_buffer]
                 
                 pitch_mean = np.mean(pitch_vals)
                 pitch_std = np.std(pitch_vals)
                 yaw_mean = np.mean(yaw_vals)
                 yaw_std = np.std(yaw_vals)
                 roll_mean = np.mean(roll_vals)
                 roll_std = np.std(roll_vals)
                 ear_mean = np.mean(ear_vals)
                 ear_std = np.std(ear_vals)
                 mar_mean = np.mean(mar_vals)
                 mar_std = np.std(mar_vals)

            input_features = [[
                features.get('pitch', 0),
                features.get('yaw', 0),
                features.get('roll', 0),
                features.get('ear_left', 0),
                features.get('ear_right', 0),
                features.get('ear', 0),      
                features.get('mar', 0),
                features.get('gaze_h', 0.5), 
                features.get('gaze_v', 0.0),
                pitch_mean, pitch_std,
                yaw_mean, yaw_std,
                roll_mean, roll_std,
                ear_mean, ear_std,
                mar_mean, mar_std
            ]]
            
            try:
                raw_probs = self.model.predict_proba(input_features)[0]
                
                if self.prob_history:
                    prev_probs = self.prob_history[-1]
                    smoothed_probs = (raw_probs * 0.3) + (prev_probs * 0.7)
                else:
                    smoothed_probs = raw_probs
                
                self.prob_history.append(smoothed_probs)
                
                # Build Probability Dictionary
                for i, prob in enumerate(smoothed_probs):
                    class_id = self.model.classes_[i]
                    name = self.id_to_name.get(class_id, f"Unknown-{class_id}")
                    prob_dict[name] = prob

                # --- Apply Overrides ---
                if override_category:
                    current_category = override_category
                    
                    for k in prob_dict:
                        prob_dict[k] = 0.05 
                        
                    prob_dict[override_category] = 1.0
                    
                    if override_category == "nodding":
                        status = "Engaged"
                        raw_score = 1.0
                    elif override_category == "drinking":
                        status = "Not Engaged"
                        raw_score = 0.0
                    elif override_category == "phone":
                        status = "Not Engaged"
                        raw_score = 0.0
                    elif override_category == "raisehand":
                        status = "Engaged"
                        raw_score = 1.0
                        
                else:
                    # Standard Model Logic
                    pred_id = np.argmax(smoothed_probs)
                    pred_class_id = self.model.classes_[pred_id]
                    
                    current_category = self.id_to_name.get(pred_class_id, "Unknown")
                    is_engaged_binary = self.binary_map.get(pred_class_id, 0)
                    confidence = smoothed_probs[pred_id]
                    
                    # Frowning Filter REMOVED (User feedback: it was killing detection)
                    
                    if is_engaged_binary == 1:
                        raw_score = 0.5 + (0.5 * confidence)
                    else:
                        raw_score = 0.5 - (0.5 * confidence)
                        
                    prob_dict["nodding"] = 0.0 

            except Exception as e:
                print(f"Prediction Error: {e}")
                raw_score = 0.0
        else:
            # print("Error: No model loaded. Cannot calculate score.")
            return 0.0, "Model Error", [], {}, {}
        
        #Update history
        self.pitch_history.append(features.get('pitch', 0))

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

        return smoothed_score, status, current_category, prob_dict

