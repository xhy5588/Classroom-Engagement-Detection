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
LABEL_MAPPING = {
    # Engaged
    "neutralface": 0,
    "frowning": 1,  # Assuming frowning indicates concentration/focus
    # "nodding": REMOVED
    "raisehand": 2,

    # Not Engaged
    # "drinking": 2,
    "phone": 3,
    "yawning": 4,
    "tilt": 5,

}

BINARY_MAP = {
    0: 1, 1: 1, 2:1,       # Engaged
    3: 0, 4: 0, 5: 0 # Not Engaged
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

    # For images, we treat them as a single frame with 0 variance
    features = extract_raw_features(image, holistic, extractor)
    if features:
        # Generate synthetic history with noise to simulate "still video"
        # This prevents the model from overfitting to "zero variance" = "not nodding"
        synthetic_history = []
        window_size = 30
        
        for _ in range(window_size):
            noisy_features = features.copy()
            # Add small Gaussian noise to temporal keys
            for key in ['pitch', 'yaw', 'roll', 'ear', 'mar']:
                if key in noisy_features:
                    # Noise scale: 0.5 degrees for angles, 0.005 for ratios
                    scale = 0.5 if key in ['pitch', 'yaw', 'roll'] else 0.005
                    noisy_features[key] += np.random.normal(0, scale)
            synthetic_history.append(noisy_features)

        # Add temporal features (std > 0 now)
        feature_vector = create_feature_vector(features, synthetic_history, is_video=True)
        data_list.append(feature_vector)
        label_list.append(label)

def process_video(video_path, holistic, extractor, label, data_list, label_list):
    cap = cv2.VideoCapture(video_path)
    
    # Buffer for sliding window
    window_size = 30
    feature_buffer = []
    
    frame_count = 0
    sample_rate = 5 # Add to dataset every 5 frames
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        features = extract_raw_features(frame, holistic, extractor)
        if features:
            feature_buffer.append(features)
            
            # Maintain buffer size
            if len(feature_buffer) > window_size:
                feature_buffer.pop(0)
            
            # Only add to dataset if we have enough history AND it's a sampling frame
            if len(feature_buffer) >= window_size and frame_count % sample_rate == 0:
                # Calculate temporal stats
                feature_vector = create_feature_vector(features, feature_buffer, is_video=True)
                data_list.append(feature_vector)
                label_list.append(label)
                
        frame_count += 1
        
    cap.release()

def extract_raw_features(image, holistic, extractor):
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    
    if results.face_landmarks:
        return extractor.extract_features(
            results.face_landmarks, 
            results.pose_landmarks, 
            image.shape
        )
    return None

def create_feature_vector(current_features, history, is_video=True):
    # Base features
    vector = {
        'pitch': current_features['pitch'],
        'yaw': current_features['yaw'],
        'roll': current_features['roll'],
        'ear_left': current_features['ear_left'],
        'ear_right': current_features['ear_right'],
        'ear_avg': current_features['ear'],
        'mar': current_features['mar'],
        'gaze_h': current_features.get('gaze_h', 0.5), 
        'gaze_v': current_features.get('gaze_v', 0.0)
    }
    
    # Temporal features to calculate stats for
    temporal_keys = ['pitch', 'yaw', 'roll', 'ear', 'mar']
    
    if is_video and isinstance(history, list):
        # Calculate stats from history buffer
        for key in temporal_keys:
            values = [f[key] for f in history]
            vector[f'{key}_mean'] = np.mean(values)
            vector[f'{key}_std'] = np.std(values)
    else:
        # Static image or single frame fallback
        for key in temporal_keys:
            val = current_features[key] if key in current_features else current_features.get('ear_avg' if key == 'ear' else key, 0)
            # Note: 'ear' key in features is 'ear' (avg), but we mapped it to 'ear_avg' in vector previously. 
            # In extract_features it returns 'ear'.
            
            vector[f'{key}_mean'] = val
            vector[f'{key}_std'] = 0.0
            
    return vector

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
    # # Balance the dataset (Undersample Majority)
    # g = df.groupby('label')
    # try:
    #     df = g.apply(lambda x: x.sample(g.size().min()), include_groups=False).reset_index(drop=True)
    #     # Restore label column if lost during include_groups=False
    #     if 'label' not in df.columns:
    #          # Re-merge or handle depending on pandas version, 
    #          # simpler fix for older pandas compat:
    #          df = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    # except TypeError:
    #      df = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    # print("Balanced Dataset Counts:\n", df['label'].value_counts())

    # Skip balancing - use unbalanced data as-is
    print("Using unbalanced dataset (no undersampling applied)")
    print("Class distribution:\n", df['label'].value_counts())
    

    # 2. Prepare Data for Training
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Model
    # RandomForest is excellent for this: handles non-linear relationships well (e.g., yaw vs engagement)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    sorted_labels = sorted(LABEL_MAPPING.items(), key=lambda item: item[1])
    target_names = [item[0] for item in sorted_labels]
    
    # Check if all classes are present in y_test to avoid sklearn errors
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    filtered_names = [target_names[i] for i in unique_labels]

    print(classification_report(y_test, y_pred, target_names=filtered_names))
    
    # 5. Save Model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    id_to_name = {v: k for k, v in LABEL_MAPPING.items()}
    
    payload = {
        "model": model,
        "id_to_name": id_to_name,
        "binary_map": BINARY_MAP
    }
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print("Done!")

if __name__ == "__main__":
    train_and_save()