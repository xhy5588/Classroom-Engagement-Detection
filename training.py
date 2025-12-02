import os
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.visual_feature_extractor import VisualFeatureExtractor

# --- CONFIGURATION ---
DATASET_PATH = "data"  # Root folder containing category subfolders
MODEL_SAVE_PATH = "engagement_model.pkl"

# Map your specific raw labels to Engagement Score (0.0 = Not Engaged, 1.0 = Engaged)
# We will now train on the KEYS (Activities) and map to VALUES (Engagement) during inference.
ACTIVITY_TO_ENGAGEMENT = {
    # Engaged
    "neutralface": 1.0,
    "frowning": 1.0,
    "nodding": 1.0, # Added nodding
    
    # Not Engaged
    "drinking": 0.0,
    "phone": 0.0,
    "yawning": 0.0,
    "tilt": 0.0,
    "raisehand": 1.0, # Changed to 1.0 as raising hand is participation
    "watch": 0.0
}

def extract_data_from_dataset(dataset_path):
    """
    Iterates through folders, processes videos/images, extracts features, 
    and returns a DataFrame.
    """
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    extractor = VisualFeatureExtractor()
    
    data = []
    labels = []
    
    print(f"Scanning dataset at: {dataset_path}")
    
    # Context manager for Holistic model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        static_image_mode=False) as holistic:
        
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            
            # Skip if not a directory or not in our mapping
            if not os.path.isdir(category_path):
                continue
            if category not in ACTIVITY_TO_ENGAGEMENT:
                print(f"Skipping unknown category folder: {category}")
                continue
                
            # Use the CATEGORY NAME as the label for Multi-class classification
            target_label = category
            print(f"Processing category: {category}")
            
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                ext = os.path.splitext(filename)[1].lower()
                
                # Handle Video Files
                if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    process_video(file_path, holistic, extractor, target_label, data, labels)
                    
                # Handle Image Files
                elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    process_image(file_path, holistic, extractor, target_label, data, labels)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels
    return df

from collections import deque

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

def process_video(video_path, holistic, extractor, label, data_list, label_list):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    skip_frames = 5 # Process every 5th frame to reduce redundancy and speed up training
    
    # Initialize buffer for this video
    feature_buffer = FeatureBuffer(maxlen=30) # ~1 second window
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        # We process every frame to update buffer, but might only save data every N frames
        # Actually, for temporal features, we need to process every frame to fill the buffer correctly.
        # But we can choose to SAVE the training sample only every N frames.
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        if results.face_landmarks:
            features = extractor.extract_features(
                results.face_landmarks, 
                results.pose_landmarks, 
                frame.shape
            )
            
            if features:
                feature_buffer.add(features)
                
                # Save sample every 'skip_frames'
                if frame_count % skip_frames == 0:
                    stats = feature_buffer.get_stats()
                    if stats:
                        # Combine raw features (optional) or just use stats?
                        # Let's use stats + current frame features (maybe)
                        # For robustness, let's use stats primarily for temporal things.
                        # But let's keep it simple: Just add all stats to the data list.
                        data_list.append(stats)
                        label_list.append(label)
        
    cap.release()

def process_image(image_path, holistic, extractor, label, data_list, label_list):
    # Images don't have temporal context, so std = 0
    image = cv2.imread(image_path)
    if image is None: 
        return

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    
    if results.face_landmarks:
        features = extractor.extract_features(
            results.face_landmarks, 
            results.pose_landmarks, 
            image.shape
        )
        
        if features:
            # Create a dummy buffer with just this one frame
            feature_buffer = FeatureBuffer(maxlen=1)
            feature_buffer.add(features)
            stats = feature_buffer.get_stats()
            
            # Manually set std to 0 (already handled by get_stats logic for len=1, but good to ensure)
            data_list.append(stats)
            label_list.append(label)

# Removed original process_frame as it is now integrated into process_video/process_image

def train_and_save():
    # 1. Extract Data
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset folder '{DATASET_PATH}' not found.")
        print("Please create the folder structure described in the guide.")
        return
    print("Starting Feature Extraction...")
    df = extract_data_from_dataset(DATASET_PATH)
    
    if df.empty:
        print("No data found! Check your dataset structure.")
        return

    print(f"Extracted {len(df)} samples.")
    print(df['label'].value_counts())

    # 2. Prepare Data for Training
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Model
    # RandomForest is excellent for this: handles non-linear relationships well (e.g., yaw vs engagement)
    print("Training Random Forest Classifier (Multi-class)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Done!")

if __name__ == "__main__":
    train_and_save()