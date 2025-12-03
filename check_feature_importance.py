import pickle
import pandas as pd
import numpy as np

def check_importance():
    try:
        with open("engagement_model.pkl", "rb") as f:
            model = pickle.load(f)
            
        print("Model loaded successfully.")
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # We need the feature names. 
            # They are not saved in the model object by default unless we used a specific sklearn version/method, 
            # but we know the order from training.py
            
            # Base cols from training.py
            cols = ['pitch', 'yaw', 'roll', 'ear_left', 'ear_right', 'ear', 'mar', 'gaze_h', 'gaze_v']
            feature_names = []
            for col in cols:
                feature_names.append(f'{col}_mean')
                feature_names.append(f'{col}_std')
            feature_names.append('pitch_freq')
            feature_names.append('pitch_energy')
            
            # Create DataFrame
            feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feat_imp = feat_imp.sort_values(by='importance', ascending=False)
            
            print("\n--- Feature Importances ---")
            print(feat_imp)
            
        else:
            print("Model does not have feature_importances_ attribute.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_importance()
