# Engagement Detection Model Plan

## 1. Project Goal
Develop a Machine Learning model to estimate an "Engagement Score" (e.g., 0-1 or 1-10) for participants in a Zoom-like video conference setting. The model analyzes visual cues to determine if a person is paying attention.

## 2. Feature Engineering (What to measure)
Instead of feeding raw pixels to a neural network (which requires massive datasets), we will use MediaPipe to extract semantic features.

### Key Visual Indicators of Engagement:
1.  **Head Pose (Orientation):** Is the face directed toward the screen?
    *   *Engaged:* Facing forward (low yaw/pitch angles).
    *   *Distracted:* Looking sideways, up, or down.
2.  **Gaze Estimation (Iris Tracking):** Are the eyes looking at the content?
    *   *Engaged:* Iris centered or scanning screen area.
    *   *Distracted:* Eyes looking off-screen (phone, window).
3.  **Eye State (Blinking/Drowsiness):**
    *   *Engaged:* Normal blink rate.
    *   *Distracted/Bored:* Eyes closed (sleepy), rapid blinking (stress), or staring blankly.
    *   *Metric:* Eye Aspect Ratio (EAR).
4.  **Mouth State (Yawning/Talking):**
    *   *Distracted:* Yawning (high Mouth Aspect Ratio).
    *   *Engaged:* Neutral or slight smile.
5.  **Body Posture:**
    *   *Engaged:* Leaning slightly forward or upright.
    *   *Distracted:* Slouching heavily, leaning way back, or absent from frame.

## 3. Data Collection Strategy

### Step A: Define Scenarios
You need to record video clips of people acting out specific states.
*   **Class 0: Not Engaged / Distracted**
    *   Looking at phone (down).
    *   Looking away (side/up).
    *   Eyes closed (sleeping).
    *   Yawning continuously.
    *   Empty chair.
    *   Talking to someone off-camera.
*   **Class 1: Engaged / Focused**
    *   Reading text on screen.
    *   Watching a video on screen.
    *   Taking notes (glancing down briefly then back up).
    *   Nodding.

### Step B: Recording Protocol
*   **Resolution:** Standard Webcam (720p or 1080p).
*   **Lighting:** Vary lighting (bright, dim, side-lit) to make the model robust.
*   **Distance:** Vary distance from camera (close-up vs. further back).
*   **Duration:** Record 5-10 minute sessions switching between engaged and distracted states.

## 4. Data Labeling

### Method: Temporal Annotation
Since engagement changes over time, you cannot label a whole video as "Engaged". You must label time segments.

**Tool Recommendation:**
*   **Simple:** Rename files like `engaged_01.mp4`, `distracted_01.mp4` and trim them strictly.
*   **Advanced:** Use a tool like **CVAT** (Computer Vision Annotation Tool) or **LabelImg** to mark start/end times of engagement.

**Label Format:**
We will likely use a **Binary Classification** (0 or 1) or a **3-Class System** (0=Distracted, 0.5=Passive, 1=Highly Engaged). Let's start with Binary (0/1).

## 5. Data Format for Training

We will process the video into a tabular format (CSV) to train a lightweight Classifier (like Random Forest, XGBoost, or a simple Neural Net).

### The `dataset.csv` Structure:
Each row represents **one frame** (or a small window of frames).

| Column Name | Description | Data Type |
| :--- | :--- | :--- |
| `timestamp` | Time in video | Float |
| `head_yaw` | Horizontal head rotation (degrees) | Float |
| `head_pitch` | Vertical head rotation (degrees) | Float |
| `head_roll` | Tilt head rotation (degrees) | Float |
| `gaze_score_x` | Horizontal eye direction (normalized) | Float |
| `gaze_score_y` | Vertical eye direction (normalized) | Float |
| `ear_left` | Eye Aspect Ratio (Left) - measures openness | Float |
| `ear_right` | Eye Aspect Ratio (Right) | Float |
| `mar` | Mouth Aspect Ratio - measures yawning | Float |
| `posture_offset_y` | Vertical nose position change (slouching) | Float |
| **`label`** | **0 (Distracted) or 1 (Engaged)** | **Integer** |

## 6. Implementation Roadmap

1.  **Data Recorder Script:** A Python script to record webcam video and save it to a folder.
2.  **Feature Extractor Script:** A script that:
    *   Reads the recorded videos.
    *   Runs MediaPipe Face Mesh.
    *   Calculates the math for Pitch/Yaw/Roll, EAR, MAR.
    *   Saves the results to `training_data.csv` with the label derived from the filename (e.g., if filename is `distracted_01.mp4`, label=0).
3.  **Model Training:**
    *   Load `training_data.csv`.
    *   Split into Train/Test sets.
    *   Train a `RandomForestClassifier` or LSTM (if using sequences).
    *   Save the model (e.g., `engagement_model.pkl`).
4.  **Real-time Inference:**
    *   Live webcam feed -> Extract Features -> Predict using Model -> Display Score.

