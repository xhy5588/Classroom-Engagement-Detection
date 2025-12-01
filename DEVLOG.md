# 🛠️ Developer Log & Quick Start Guide

**Current Version:** Demo v0.5 (Engagement Logic Implemented)  
**Date:** Dec 2, 2025

---

## 👋 Welcome Team!
This document tracks our progress and explains how to play with the current codebase. We are building a **Classroom Engagement Detection System** using Computer Vision.

## 🚀 Quick Start: Test the Full System
We have a working system that calculates an **Engagement Score** in real-time.

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```
*(Make sure you have `mediapipe`, `opencv-python`, and `numpy`)*

**2. Run the Demo:**
```bash
python test_engagement_system.py
```

**3. Controls:**
*   Press **`q`** to quit the window.

---

## 📊 What am I looking at?
The system displays your **Status** (Engaged/Distracted) and detects specific behaviors.

| Behavior Label | Trigger Condition | How to test it |
| :--- | :--- | :--- |
| **Looking Away** | `yaw > 30°` | Turn your head left or right significantly. |
| **Looking Down/Phone** | `pitch < -25°` | Look down at your lap (as if checking a phone). |
| **Yawning** | `mar > 0.5` | Open your mouth wide. |
| **Eyes Closed** | `ear < 0.15` | Close your eyes for a second (simulating sleep). |
| **Nodding** | `pitch` variance | Nod your head up and down continuously. |
| **Head Tilt** | `roll > 20°` | Tilt your head sideways (resting on hand). |

*(Note: Hand Raise detection was removed due to false positives with eating/phone use.)*

---

## 📁 Project Structure
*   `src/visual_feature_extractor.py` -> **The Eyes**. Extracts raw numbers (angles, ratios) from the video.
*   `src/engagement_scorer.py` -> **The Brain**. Decides if you are engaged based on those numbers.
*   `test_engagement_system.py` -> **The Demo**. Runs the webcam and shows the UI.

## 📝 Current Status
- [x] **Face Detection**: Working (MediaPipe Face Mesh)
- [x] **Head Pose (Yaw/Pitch/Roll)**: Working
- [x] **Blink Detection (EAR)**: Working
- [x] **Yawn Detection (MAR)**: Working
- [x] **Engagement Logic**: Implemented (Simple Heuristic Rules)
- [x] **Smoothing**: Score doesn't jitter wildy
- [ ] **Hand Raise**: *Removed temporarily* (Needs better logic)

## 🔜 What's Next?
1.  **Data Collection**: We need to record video clips to validate if these rules are accurate.
2.  **Tuning**: Adjusting the sensitivity (e.g., is "Looking Down" triggering too easily?).
3.  **Dashboard**: Building a prettier interface for the teacher.

---

## 💡 Tips for Collaborators
*   **Lighting Matters**: If the room is too dark, MediaPipe might lose your face.
*   **Distance**: Sit 0.5m - 1m away from the webcam for best results.
*   **Privacy**: Everything runs LOCALLY on your computer. No video is sent to the cloud.
