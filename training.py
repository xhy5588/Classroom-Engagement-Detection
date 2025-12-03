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

# Map your specific raw labels to Binary Engagement (0 = Not Engaged, 1 = Engaged)
# Map your specific raw labels to Class Names
LABEL_MAPPING = {
    # Engaged
    "neutralface": "neutral",
    "frowning": "frowning",
    # "nodding": "nodding", # REMOVED: Unstable detection
    "raisehand": "raisehand",

    # Not Engaged
    "drinking": "drinking",
    "phone": "phone",
    "yawning": "yawning",
    "tilt": "tilt",
    "watch": "watch",
    # "sleeping": "sleeping" # If you have this folder
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
            if category not in LABEL_MAPPING:
                print(f"Skipping unknown category folder: {category}")
                continue
                
            target_label = LABEL_MAPPING[category]
            print(f"Processing category: {category} (Target: {target_label})")
            
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

def process_image(image_path, holistic, extractor, label, data_list, label_list):
    image = cv2.imread(image_path)
    if image is None: 
        return

    # Treat image as a static video of WINDOW_SIZE length
    # Just extract once and replicate stats (std will be 0)
    # Actually, let's just use the same process_frame logic but wrapped.
    
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
            # Create a fake buffer of identical frames
            # Mean = value, Std = 0
            stats = {}
            cols = ['pitch', 'yaw', 'roll', 'ear_left', 'ear_right', 'ear', 'mar', 'gaze_h', 'gaze_v']
            for col in cols:
                val = features.get(col, 0)
                stats[f'{col}_mean'] = val
                stats[f'{col}_std'] = 0.0
            
            data_list.append(stats)
            label_list.append(label)

def process_video(video_path, holistic, extractor, label, data_list, label_list):
    cap = cv2.VideoCapture(video_path)
    
    # Windowing Configuration
    WINDOW_SIZE = 30 # ~1 second at 30fps
    STRIDE = 15      # 50% overlap
    
    frame_buffer = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Extract features for this frame
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
                frame_buffer.append(features)
                
        # Process Buffer if full
        if len(frame_buffer) >= WINDOW_SIZE:
            # Calculate Window Stats
            window_data = calculate_window_stats(frame_buffer)
            data_list.append(window_data)
            label_list.append(label)
            
            # Slide Window
            frame_buffer = frame_buffer[STRIDE:]
        
    cap.release()

def calculate_window_stats(buffer):
    """Calculates Mean and Std Dev for a buffer of features."""
    df = pd.DataFrame(buffer)
    stats = {}
    
    # Features to aggregate
    cols = ['pitch', 'yaw', 'roll', 'ear_left', 'ear_right', 'ear', 'mar', 'gaze_h', 'gaze_v']
    
    for col in cols:
        if col in df.columns:
            stats[f'{col}_mean'] = df[col].mean()
            stats[f'{col}_std'] = df[col].std()
            
    return stats

def train_and_save():
    # 1. Extract Data
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset folder '{DATASET_PATH}' not found.")
        print("Please create the folder structure described in the guide.")
        return
    print("Starting Feature Extraction (Windowed)...")
    df = extract_data_from_dataset(DATASET_PATH)
    
    if df.empty:
        print("No data found! Check your dataset structure.")
        return

    print(f"Extracted {len(df)} samples.")
    print(df['label'].value_counts())
    
    # REMOVED UNDERSAMPLING to keep all data
    # We will use class_weight='balanced' instead
    
    # 2. Prepare Data for Training
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Handle NaNs (if any std dev calculation failed or empty columns)
    X = X.fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train Model
    print("Training Random Forest Classifier (Balanced)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
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
            if category not in LABEL_MAPPING:
                print(f"Skipping unknown category folder: {category}")
                continue
                
            target_label = LABEL_MAPPING[category]
            print(f"Processing category: {category} (Target: {target_label})")
            
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

def process_image(image_path, holistic, extractor, label, data_list, label_list):
    image = cv2.imread(image_path)
    if image is None: 
        return

    process_frame(image, holistic, extractor, label, data_list, label_list)

def process_video(video_path, holistic, extractor, label, data_list, label_list):
    cap = cv2.VideoCapture(video_path)
    
    # Windowing Configuration
    WINDOW_SIZE = 30 # ~1 second at 30fps
    STRIDE = 15      # 50% overlap
    
    frame_buffer = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Extract features for this frame
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
                frame_buffer.append(features)
                
        # Process Buffer if full
        if len(frame_buffer) >= WINDOW_SIZE:
            # Calculate Window Stats
            window_data = calculate_window_stats(frame_buffer)
            data_list.append(window_data)
            label_list.append(label)
            
            # Slide Window
            frame_buffer = frame_buffer[STRIDE:]
        
    cap.release()

def calculate_window_stats(buffer):
    """Calculates Mean and Std Dev for a buffer of features."""
    df = pd.DataFrame(buffer)
    stats = {}
    
    # Features to aggregate
    cols = ['pitch', 'yaw', 'roll', 'ear_left', 'ear_right', 'ear', 'mar', 'gaze_h', 'gaze_v',
            'hand_to_mouth', 'hand_to_ear', 'hand_height', 'brow_eye_dist', 'mouth_curvature']
    
    for col in cols:
        if col in df.columns:
            stats[f'{col}_mean'] = df[col].mean()
            stats[f'{col}_std'] = df[col].std()
            
    return stats

# Note: process_image needs to be updated or removed if we strictly require temporal windows.
# For images, we can duplicate the single frame to fill the window (static behavior).
def process_image(image_path, holistic, extractor, label, data_list, label_list):
    image = cv2.imread(image_path)
    if image is None: 
        return

    # Treat image as a static video of WINDOW_SIZE length
    # Just extract once and replicate stats (std will be 0)
    # Actually, let's just use the same process_frame logic but wrapped.
    
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
            # Create a fake buffer of identical frames
            # Mean = value, Std = 0
            stats = {}
            cols = ['pitch', 'yaw', 'roll', 'ear_left', 'ear_right', 'ear', 'mar', 'gaze_h', 'gaze_v',
                    'hand_to_mouth', 'hand_to_ear', 'hand_height', 'brow_eye_dist', 'mouth_curvature']
            for col in cols:
                val = features.get(col, 0)
                stats[f'{col}_mean'] = val
                stats[f'{col}_std'] = 0.0
            
            data_list.append(stats)
            label_list.append(label)

def train_and_save():
    # 1. Extract Data
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset folder '{DATASET_PATH}' not found.")
        print("Please create the folder structure described in the guide.")
        return
    print("Starting Feature Extraction (Windowed)...")
    df = extract_data_from_dataset(DATASET_PATH)
    
    if df.empty:
        print("No data found! Check your dataset structure.")
        return

    print(f"Extracted {len(df)} samples.")
    print(df['label'].value_counts())
    
    # 2. Prepare Data for Training
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Handle NaNs (if any std dev calculation failed or empty columns)
    X = X.fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train Model
    print("Training Random Forest Classifier (Balanced)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
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