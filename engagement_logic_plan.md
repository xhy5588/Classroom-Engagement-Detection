# Engagement Logic Plan (Draft)

## 1. Goal
Convert raw visual features into a single **Engagement Score (0.0 to 1.0)** and a **Status Label** (e.g., "Engaged", "Distracted").

## 2. Input Features
The logic receives a dictionary for every video frame:
*   `pitch`, `yaw`, `roll` (Head angles)
*   `ear` (Eye Aspect Ratio - open/closed)
*   `mar` (Mouth Aspect Ratio - open/closed)
*   `hand_raised` (Boolean)
*   `gaze_score` (To be added: simple check if looking at screen center)

## 3. Logic Rules (Heuristics)

We will use a **Weighted Scoring System**. Each behavior contributes positively or negatively to the total score.

### A. Negative Factors (Distractions)
| Behavior | Condition (Thresholds need tuning) | Penalty |
| :--- | :--- | :--- |
| **Looking Away** | `abs(yaw) > 30°` OR `abs(pitch) > 25°` | -0.4 |
| **Drowsy / Sleeping** | `ear < 0.15` (Eyes closed) for > 1 sec | -0.8 |
| **Yawning** | `mar > 0.5` (Mouth wide open) | -0.3 |
| **Head Tilt** | `abs(roll) > 25°` (Slouching/resting head) | -0.2 |

### B. Positive Factors (Engagement)
| Behavior | Condition | Bonus |
| :--- | :--- | :--- |
| **Looking at Screen** | `abs(yaw) < 15°` AND `abs(pitch) < 15°` | +0.1 |
| **Hand Raise** | `hand_raised == True` | Sets Score to **1.0** (Max) |
| **Active Speaking** | `mar > 0.1` AND `mar < 0.4` (Talking) | +0.2 |

## 4. Scoring Formula

```python
base_score = 0.5 # Start neutral

# Apply penalties and bonuses
current_score = base_score - penalty_looking_away - penalty_drowsy + bonus_looking_at_screen

# Clamp result
final_score = max(0.0, min(1.0, current_score))
```

## 5. Smoothing (Temporal Consistency)
A single frame of "looking away" shouldn't instantly tank the score.
*   **Logic**: We will use an **Exponential Moving Average (EMA)**.
*   `smoothed_score = (current_score * alpha) + (previous_score * (1 - alpha))`
*   This makes the score stable and less "jittery".

## 6. Output Classes
Based on `final_score`:
*   **> 0.7**: `Highly Engaged` (Green)
*   **0.3 - 0.7**: `Passive / Neutral` (Yellow)
*   **< 0.3**: `Distracted` (Red)

## 7. Implementation Structure
We will create a new file: `src/engagement_scorer.py`.

```python
class EngagementScorer:
    def __init__(self):
        self.history = []
        
    def calculate_score(self, features):
        # 1. Check thresholds
        # 2. Calculate raw score
        # 3. Smooth score over time
        # 4. Return score + status label
```

---
**Please review:**
1.  Are the thresholds (30 degrees, etc.) reasonable?
2.  Do you want to prioritize any specific behavior (like Hand Raising)?
3.  Should "Yawning" be a heavy penalty or just a minor one?

