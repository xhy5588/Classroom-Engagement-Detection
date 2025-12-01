import numpy as np
from collections import deque

class EngagementScorer:
    def __init__(self):
        # History buffers for smoothing
        self.score_history = deque(maxlen=30) # 1 second at 30fps
        self.pitch_history = deque(maxlen=10)
        
        # State tracking for behaviors
        self.is_nodding = False
        self.is_speaking = False
        
        # Scoring Weights
        self.base_score = 0.5
        self.weights = {
            # 'hand_raise': 1.0,     # Removed due to false positives
            'looking_away': -0.3,
            'phone_use': -0.4,
            'yawning': -0.3,
            'sleeping': -0.8,
            'nodding': 0.2,
            'speaking': 0.15
        }

    def calculate_score(self, features):
        """
        Calculate engagement score (0.0 to 1.0) based on visual features.
        Returns: (score, status_text, detected_behaviors)
        """
        if not features:
            return 0.0, "No Face", []

        score = self.base_score
        behaviors = []

        # Unpack features
        pitch = features.get('pitch', 0)
        yaw = features.get('yaw', 0)
        roll = features.get('roll', 0)
        ear = features.get('ear', 0)
        mar = features.get('mar', 0)
        # hand_raised = features.get('hand_raised', False) # Removed

        # --- Behavior Detection ---

        # 1. Hand Raise (Removed)
        # if hand_raised:
        #     score = 1.0
        #     behaviors.append("Hand Raised")
        #     return score, "Highly Engaged", behaviors

        # 2. Looking at Phone (Assumption: Looking down significantly)
        # Threshold: Pitch < -25 degrees (looking down)
        if pitch < -25:
            score += self.weights['phone_use']
            behaviors.append("Looking Down/Phone")
        
        # 3. Looking Away (Side)
        elif abs(yaw) > 30:
            score += self.weights['looking_away']
            behaviors.append("Looking Away")
        
        # 4. Yawning
        # Threshold: MAR > 0.5 (Open wide)
        if mar > 0.5:
            score += self.weights['yawning']
            behaviors.append("Yawning")
            
        # 5. Sleeping / Drowsy
        # Threshold: EAR < 0.15 (Eyes closed)
        if ear < 0.15:
            score += self.weights['sleeping']
            behaviors.append("Eyes Closed")

        # 6. Head Tilt
        if abs(roll) > 20:
             behaviors.append(f"Head Tilt {'Left' if roll > 0 else 'Right'}")

        # 7. Nodding Detection (Simple Variance check)
        self.pitch_history.append(pitch)
        if len(self.pitch_history) == 10:
            pitch_variance = np.var(self.pitch_history)
            # If variance is high but mean is roughly centered, likely nodding
            if pitch_variance > 10 and abs(np.mean(self.pitch_history)) < 15:
                score += self.weights['nodding']
                behaviors.append("Nodding")

        # 8. Speaking Detection (Visual only)
        # Rapid changes in MAR usually indicate speaking
        # (For now, we use a simple threshold range)
        if 0.1 < mar < 0.4:
             # This is weak without audio, but a placeholder
             pass

        # --- Final Score Calculation ---
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        
        # Smoothing (Exponential Moving Average)
        if self.score_history:
            prev_score = self.score_history[-1]
            smoothed_score = (score * 0.2) + (prev_score * 0.8)
        else:
            smoothed_score = score
            
        self.score_history.append(smoothed_score)

        # Status Text
        if smoothed_score > 0.7:
            status = "Engaged"
        elif smoothed_score > 0.3:
            status = "Neutral"
        else:
            status = "Distracted"

        return smoothed_score, status, behaviors

