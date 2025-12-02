import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from src.visual_feature_extractor import VisualFeatureExtractor
from collections import deque

# Re-use FeatureBuffer logic
class FeatureBuffer:
    def __init__(self, maxlen=30):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, features):
        self.buffer.append(features)
        
    def get_stats(self):
        if not self.buffer:
            return {}
        df = pd.DataFrame(list(self.buffer))
        stats = {}
        for col in ['pitch', 'yaw', 'roll', 'ear', 'mar', 'hand_face_dist']:
            if col in df.columns:
                stats[f'{col}_mean'] = df[col].mean()
                stats[f'{col}_std'] = df[col].std() if len(df) > 1 else 0.0
        return stats

def check_stats():
    dataset_path = "data"
    extractor = VisualFeatureExtractor()
    mp_holistic = mp.solutions.holistic
    
    data_stats = []

    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path) or category == "labeling_support":
                continue
                
            print(f"Processing {category}...")
            
            file_count = 0
            for file in os.listdir(category_path):
                if not file.endswith(".mp4"):
                    continue
                if file_count >= 2:
                    break
                file_count += 1
                    
                video_path = os.path.join(category_path, file)
                cap = cv2.VideoCapture(video_path)
                buffer = FeatureBuffer(maxlen=30)
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                        
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    
                    if results.face_landmarks:
                        features = extractor.extract_features(results.face_landmarks, results.pose_landmarks, frame.shape)
                        if features:
                            buffer.add(features)
                            stats = buffer.get_stats()
                            if 'pitch_std' in stats:
                                data_stats.append({
                                    'category': category,
                                    'pitch_std': stats['pitch_std'],
                                    'pitch_mean': stats['pitch_mean']
                                })
                cap.release()

    df = pd.DataFrame(data_stats)
    print("\n--- Training Data Statistics ---")
    print(df.groupby('category')['pitch_std'].describe())

if __name__ == "__main__":
    check_stats()
