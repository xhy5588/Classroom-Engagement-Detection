import pandas as pd
import numpy as np
import os
import cv2
import mediapipe as mp
from src.visual_feature_extractor import VisualFeatureExtractor
from training import calculate_window_stats

# Configuration
DATASET_PATH = "data"
CLASSES_TO_COMPARE = ["neutralface", "nodding"] # neutralface maps to neutral

def analyze_distribution():
    print("Analyzing Feature Distribution...")
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    extractor = VisualFeatureExtractor()
    
    feature_data = {c: [] for c in CLASSES_TO_COMPARE}
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, static_image_mode=False) as holistic:
        for category in CLASSES_TO_COMPARE:
            path = os.path.join(DATASET_PATH, category)
            if not os.path.exists(path):
                print(f"Warning: {path} not found")
                continue
                
            print(f"Processing {category}...")
            files = os.listdir(path)[:5] # Analyze first 5 videos per class
            
            for filename in files:
                filepath = os.path.join(path, filename)
                cap = cv2.VideoCapture(filepath)
                
                buffer = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    
                    if results.face_landmarks:
                        feats = extractor.extract_features(results.face_landmarks, results.pose_landmarks, frame.shape)
                        if feats: buffer.append(feats)
                        
                    if len(buffer) >= 30:
                        stats = calculate_window_stats(buffer)
                        feature_data[category].append(stats)
                        buffer = [] # Reset or slide? Let's reset for distinct samples
                        
                cap.release()
                
    # Print Stats
    for cat, samples in feature_data.items():
        if not samples: continue
        df = pd.DataFrame(samples)
        print(f"\n--- {cat} Stats ({len(df)} samples) ---")
        print(df[['pitch_mean', 'pitch_std', 'pitch_freq', 'pitch_energy', 'roll_mean']].describe().T[['mean', 'std']])

if __name__ == "__main__":
    analyze_distribution()
