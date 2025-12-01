# Demo v0 & v1 Implementation Plan
## Classroom Engagement Detection System

## Overview
This plan outlines the development of a real-time classroom engagement detection system with two demo versions:
- **Demo v0**: Real-time individual participant tracking (visual-only: facial expressions + engagement indicators)
- **Demo v1**: Aggregated dashboard showing overall audience sentiment over time

**Note**: This implementation focuses on **visual features only** (no audio processing).

---

## 🎯 Demo v0: Individual Participant Real-Time Tracking

### Goal
Track and display real-time reactions of individual participants showing:
- Live facial expression classification
- Visual engagement indicators (head pose, eye state, gaze direction, etc.)
- Real-time engagement score based on visual cues

### Phase 1: Visual Features (Facial Expression Classification)
**Timeline: Week 1-2**

#### 1.1 Facial Expression Detection
- **Technology**: MediaPipe Face Mesh (already tested) + BlendShapes
- **Expressions to Classify**:
  - Engaged: Neutral, Smiling, Concentrated
  - Distracted: Yawning, Looking away, Eyes closed
  - Confused: Frowning, Raised eyebrows
  
#### 1.2 Feature Extraction Module
**File: `src/visual_feature_extractor.py`**
- Extract facial landmarks using MediaPipe Face Mesh
- Calculate facial expression features:
  - **Eye Aspect Ratio (EAR)**: Blink detection, drowsiness
  - **Mouth Aspect Ratio (MAR)**: Yawning detection
  - **Head Pose Angles** (Yaw, Pitch, Roll): Gaze direction
  - **Facial BlendShapes**: Direct expression classification (smile, frown, etc.)
  - **Gaze Estimation**: Where person is looking

**Features Output per frame:**
```python
{
    'ear_left': float,      # Eye Aspect Ratio (left)
    'ear_right': float,     # Eye Aspect Ratio (right)
    'mar': float,           # Mouth Aspect Ratio
    'head_yaw': float,      # Horizontal head rotation
    'head_pitch': float,    # Vertical head rotation
    'head_roll': float,     # Tilt rotation
    'gaze_x': float,        # Gaze direction X
    'gaze_y': float,        # Gaze direction Y
    'expression': str,      # Top expression class
    'expression_score': float,
    'blinks_per_minute': float,
    'is_yawn': bool
}
```

#### 1.3 Expression Classification
- **Option A**: Use MediaPipe BlendShapes directly (48 facial action units)
- **Option B**: Train simple classifier on extracted features
- **Initial Approach**: Start with Option A (off-the-shelf), validate, then refine

**Expression Categories:**
1. **High Engagement**: Smile, Concentrated look, Active listening
2. **Medium Engagement**: Neutral, Passive
3. **Low Engagement**: Yawn, Eyes closed, Looking away, Distracted

---

### Phase 2: Real-Time Processing Pipeline
**Timeline: Week 2-3**

#### 2.1 Real-Time Video Processing
**File: `src/realtime_processor.py`**
- Process video frames at 30 FPS
- MediaPipe Face Mesh processing per frame
- Feature extraction and expression classification
- Frame dropping strategy if processing lags

#### 2.2 Demo v0 Application
**File: `demo_v0_individual_tracker.py`**

**Display Layout:**
```
┌─────────────────────────────────────────┐
│  Live Video Feed                         │
│  [Face landmarks overlay]                │
│  [Expression label]                      │
├─────────────────────────────────────────┤
│  Real-Time Visual Metrics                │
│  - Expression: [Smiling] (0.85)          │
│  - Head Pose: Forward (✓)                │
│    • Yaw: 5°  Pitch: -2°  Roll: 1°      │
│  - Eye State: Open (✓)                   │
│    • Blinks: 15/min (normal)             │
│    • EAR Left: 0.28  Right: 0.29        │
│  - Gaze: On-screen (✓)                   │
│    • Gaze X: 0.48  Y: 0.52              │
│  - Mouth: Neutral (✓)                    │
│    • MAR: 0.15  (no yawn)               │
│  - Engagement Score: 0.82                │
├─────────────────────────────────────────┤
│  Time Series Graph (last 30 seconds)     │
│  [Engagement score over time]            │
└─────────────────────────────────────────┘
```

**Engagement Score Calculation (v0 - Visual Only):**
```python
engagement_score = (
    0.35 * expression_score +     # Facial expression weight
    0.25 * head_pose_score +      # Looking at screen
    0.20 * eye_state_score +      # Eyes open, normal blinking
    0.15 * gaze_score +           # Gaze direction on screen
    0.05 * blink_rate_score       # Normal blink rate
)
```

---

## 🎯 Demo v1: Aggregated Dashboard
**Timeline: Week 4-5**

### Goal
Show overall audience sentiment and engagement trends over time for a classroom/meeting

### Features
1. **Multi-Participant Support**: Track 4-8 participants simultaneously
2. **Aggregated Metrics**:
   - Average engagement score
   - Engagement distribution (high/medium/low counts)
   - Sentiment timeline (graph showing engagement over session)
3. **Alert System**: Notify when overall engagement drops below threshold

### Implementation
- Extend Demo v0 to handle multiple video streams
- Add participant ID tracking
- Create dashboard UI (web-based or desktop)
- Store session data for historical analysis

---

## 🔒 Privacy & Consent

### Privacy Considerations
1. **Local Processing**: All processing happens on-device (no cloud upload)
2. **Data Storage**: 
   - Option A: No storage (real-time only)
   - Option B: Store aggregated metrics only (no raw video)
3. **Consent Mechanism**:
   - **File: `src/consent_manager.py`**
   - Display consent form before starting
   - Log consent timestamp
   - Allow participants to opt-out mid-session

### Implementation
```python
# Consent form should include:
- What data is collected (video feed)
- How it's used (engagement analysis)
- Where it's stored (local/aggregated only)
- Right to opt-out
```

---

## 🧪 Testing & Validation Strategy

### Phase 1: Visual Feature Validation (Week 3-4)

#### Visual Testing
- **Diverse Lighting Conditions**:
  - Bright natural light
  - Dim indoor lighting
  - Side lighting (shadows)
  - Overhead lighting
- **Camera Angles**:
  - Front-facing (optimal)
  - Slight side angle (15-30°)
  - Higher/lower angles
- **Participant Diversity**:
  - Different skin tones
  - Glasses/no glasses
  - Facial hair variations

**Test Scenarios:**
1. Act out "engaged" behavior (reading, nodding, focused, smiling)
2. Act out "distracted" behavior (looking at phone, yawning, eyes closed, looking away)
3. Record test videos with ground truth labels
4. Calculate accuracy of expression classification
5. Validate head pose estimation accuracy
6. Test gaze estimation reliability

**Validation Metrics:**
- Expression classification accuracy (confusion matrix)
- Head pose estimation error (degrees)
- Blink detection accuracy (false positive/negative rate)
- Gaze estimation consistency
- Overall engagement score correlation with ground truth

---

## 📁 Project Structure

```
CS496-ML/
├── src/
│   ├── visual_feature_extractor.py    # Face mesh, expressions, head pose
│   ├── engagement_scorer.py           # Combine features → engagement score
│   ├── consent_manager.py             # Privacy & consent handling
│   └── utils/
│       ├── head_pose_estimator.py     # Calculate yaw/pitch/roll
│       ├── gaze_estimator.py          # Eye gaze direction
│       ├── expression_classifier.py   # Classify facial expressions
│       └── blink_detector.py          # Detect blinks and calculate EAR
├── demo_v0_individual_tracker.py      # Main Demo v0 application
├── demo_v1_dashboard.py               # Main Demo v1 application (future)
├── tests/
│   ├── test_visual_features.py
│   └── test_engagement_scoring.py
├── data/
│   ├── test_videos/                   # Validation videos
│   └── sessions/                      # Stored session data (optional)
├── requirements.txt
└── README.md
```

---

## 🚀 Implementation Steps (Week-by-Week)

### Week 1: Visual Foundation
- [ ] Extend `holistic_test.py` to extract facial expression features
- [ ] Implement head pose estimation (yaw, pitch, roll)
- [ ] Implement Eye Aspect Ratio (EAR) calculation
- [ ] Implement Mouth Aspect Ratio (MAR) calculation
- [ ] Implement gaze estimation (where person is looking)
- [ ] Create basic expression classification using BlendShapes
- [ ] Implement blink detection and tracking
- [ ] Test with different lighting conditions

### Week 2: Real-Time Pipeline
- [ ] Create real-time video processing pipeline
- [ ] Implement basic engagement score calculator (visual-only)
- [ ] Create Demo v0 UI (OpenCV display + metrics)
- [ ] Add time-series visualization (matplotlib or simple graph)
- [ ] Optimize for real-time performance (30 FPS target)

### Week 3: Validation & Testing
- [ ] Record test videos with ground truth labels
- [ ] Test visual features across diverse lighting conditions
- [ ] Test with different camera angles
- [ ] Validate expression classification accuracy
- [ ] Validate head pose and gaze estimation
- [ ] Fix bugs and improve robustness

### Week 4: Demo v1 Preparation
- [ ] Extend to multi-participant tracking
- [ ] Create aggregation logic (average scores, distributions)
- [ ] Design dashboard UI
- [ ] Implement session data storage (optional)

### Week 5: Polish & Documentation
- [ ] Add consent manager
- [ ] Final testing and optimization
- [ ] Documentation
- [ ] Prepare demo presentation

---

## 📦 Required Dependencies

```txt
# Add to requirements.txt
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0    # For visualizations
pandas>=2.0.0        # For data handling (optional, for data logging)
scikit-learn>=1.3.0  # For simple classifiers (if needed)
```

---

## 🎓 Key Design Decisions

1. **Start Simple, Iterate**: Use off-the-shelf MediaPipe models first, validate, then train custom models if needed
2. **Visual-First Approach**: Focus on robust visual feature extraction before considering audio
3. **Privacy First**: All processing local, minimal data storage
4. **Real-Time Focus**: Optimize for live demo, not batch processing (target 30 FPS)
5. **Extensible Architecture**: Design modules that can be easily extended for Demo v1 (multi-participant)

---

## 🔄 Next Steps After Demo v0

1. **Collect Real Data**: Record actual classroom/meeting sessions (with consent)
2. **Train Custom Models**: If off-the-shelf models insufficient, train on collected data
3. **Fine-tune Scoring**: Adjust engagement score weights based on validation
4. **Add More Visual Features**: Body posture, hand gestures, micro-expressions
5. **Build Dashboard**: Implement Demo v1 aggregated view
6. **Consider Audio (Future)**: Add audio features if visual-only approach needs enhancement

---

## 📝 Notes

- Keep it modular: Each feature extractor should be independent
- Document all parameters (thresholds, weights, etc.) for easy tuning
- Use configuration files for easy experimentation
- Log everything during testing for analysis

---

**Ready to start implementation? Let's begin with Week 1 tasks!**

