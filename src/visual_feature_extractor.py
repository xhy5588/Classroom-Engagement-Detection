import numpy as np
import cv2
import mediapipe as mp
import math

class VisualFeatureExtractor:
    def __init__(self):
        # MediaPipe indices for Face Mesh (Standard 468 landmarks)
        # Eyes
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # Mouth (Outer lip)
        self.MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
        
        # Face 3D reference points for Head Pose Estimation
        # Nose tip, Chin, Left Eye Left Corner, Right Eye Right Corner, Left Mouth Corner, Right Mouth Corner
        self.FACE_3D_POINTS = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # Corresponding MediaPipe Face Mesh Indices
        self.FACE_2D_INDICES = [1, 152, 263, 33, 291, 61]

    def extract_features(self, face_landmarks, pose_landmarks, image_shape):
        """
        Extracts visual features from MediaPipe face and pose landmarks.
        Returns a dictionary of features.
        """
        if not face_landmarks:
            return None

        h, w, c = image_shape
        
        # 1. Head Pose Estimation
        pitch, yaw, roll = self._get_head_pose(face_landmarks, h, w)
        
        # 2. Eye Aspect Ratio (EAR)
        ear_left = self._calculate_ear(face_landmarks, self.LEFT_EYE, h, w)
        ear_right = self._calculate_ear(face_landmarks, self.RIGHT_EYE, h, w)
        avg_ear = (ear_left + ear_right) / 2.0
        
        # 3. Mouth Aspect Ratio (MAR)
        mar = self._calculate_mar(face_landmarks, self.MOUTH, h, w)
        
        # 4. Gaze Ratio (REQUIRED for distinguishing Watch vs Neutral)
        gaze_h, gaze_v = self._calculate_gaze(face_landmarks, h, w)
        
        # 5. Hand Features (REQUIRED for Drinking, Phone, RaiseHand)
        hand_features = self._extract_hand_features(pose_landmarks, face_landmarks, h, w)
        
        # 6. Frowning Features (REQUIRED for Frowning)
        frown_features = self._extract_frown_features(face_landmarks, h, w)

        features = {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'ear_left': ear_left,
            'ear_right': ear_right,
            'ear': avg_ear,
            'mar': mar,
            'gaze_h': gaze_h,
            'gaze_v': gaze_v
        }
        
        # Merge features
        features.update(hand_features)
        features.update(frown_features)
        
        return features
        
    def _calculate_gaze(self, landmarks, h, w):
        try:
            # Left Eye: Inner(33), Outer(133), Iris(468)
            l_inner = np.array([landmarks.landmark[33].x * w, landmarks.landmark[33].y * h])
            l_outer = np.array([landmarks.landmark[133].x * w, landmarks.landmark[133].y * h])
            l_iris = np.array([landmarks.landmark[468].x * w, landmarks.landmark[468].y * h])
            
            eye_width = np.linalg.norm(l_inner - l_outer)
            if eye_width == 0: return 0.5, 0.0
            
            # Horizontal: Distance to inner corner / Total width
            dist_to_inner = np.linalg.norm(l_iris - l_inner)
            dist_to_outer = np.linalg.norm(l_iris - l_outer)
            gaze_h = dist_to_inner / (dist_to_inner + dist_to_outer)
            
            # Vertical: Iris Y relative to eye center
            center_y = (l_inner[1] + l_outer[1]) / 2
            gaze_v = (l_iris[1] - center_y) / eye_width 
            
            return gaze_h, gaze_v
        except:
            return 0.5, 0.0

    def _get_head_pose(self, landmarks, h, w):
        """
        Estimate Head Pose (Pitch, Yaw, Roll) using PnP algorithm.
        """
        face_2d = []
        
        for idx in self.FACE_2D_INDICES:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])
            
        face_2d = np.array(face_2d, dtype=np.float64)

        # Camera matrix (assumed)
        focal_length = 1 * w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ])

        # Distortion coefficients (assuming none)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(
            self.FACE_3D_POINTS, face_2d, cam_matrix, dist_matrix
        )

        if not success:
            return 0, 0, 0

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Calculate angles manually to avoid RQDecomp3x3 version issues
        # Decompose Rotation Matrix to Euler Angles (XYZ convention)
        # pitch (x), yaw (y), roll (z)
        
        sy = math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(rmat[2,1] , rmat[2,2])
            y = math.atan2(-rmat[2,0], sy)
            z = math.atan2(rmat[1,0], rmat[0,0])
        else:
            x = math.atan2(-rmat[1,2], rmat[1,1])
            y = math.atan2(-rmat[2,0], sy)
            z = 0
            
        # Convert to degrees
        pitch = x * 180 / math.pi
        yaw = y * 180 / math.pi
        roll = z * 180 / math.pi

        return pitch, yaw, roll

    def _calculate_ear(self, landmarks, indices, h, w):
        """
        Calculate Eye Aspect Ratio (EAR).
        """
        # Helper to get coordinates
        def get_point(idx):
            lm = landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])

        # Vertical distances
        v1 = np.linalg.norm(get_point(indices[1]) - get_point(indices[5]))
        v2 = np.linalg.norm(get_point(indices[2]) - get_point(indices[4]))
        
        # Horizontal distance
        h_dist = np.linalg.norm(get_point(indices[0]) - get_point(indices[3]))

        if h_dist == 0:
            return 0.0
            
        ear = (v1 + v2) / (2.0 * h_dist)
        return ear

    def _calculate_mar(self, landmarks, indices, h, w):
        """
        Calculate Mouth Aspect Ratio (MAR).
        Using outer lip points.
        """
        def get_point(idx):
            lm = landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
            
        # Vertical distances
        v1 = np.linalg.norm(get_point(indices[1]) - get_point(indices[7])) # 291-405? No, need check indices
        # Indices: [61, 291, 39, 181, 0, 17, 269, 405]
        # 0: 61 (left corner), 1: 291 (right corner)
        # 2: 39, 3: 181, 4: 0 (top), 5: 17 (bottom), 6: 269, 7: 405
        
        # Let's use a simpler set for MAR usually:
        # P1(61), P2(146? No), P3(0), P4(17), P5(291), etc.
        # Let's use basic vertical / horizontal for now using 4 points
        # Top(13), Bottom(14), Left(61), Right(291) -> Inner lips might be better for openness
        # Outer lips: Top(0), Bottom(17), Left(61), Right(291)
        
        top = get_point(0)
        bottom = get_point(17)
        left = get_point(61)
        right = get_point(291)
        
        v_dist = np.linalg.norm(top - bottom)
        h_dist = np.linalg.norm(left - right)
        
        if h_dist == 0:
            return 0.0
            
        mar = v_dist / h_dist
        mar = v_dist / h_dist
        return mar

    def _extract_hand_features(self, pose_landmarks, face_landmarks, h, w):
        """
        Extracts hand-related features:
        - hand_to_mouth_dist: Min distance from wrist to mouth (normalized)
        - hand_to_ear_dist: Min distance from wrist to ear (normalized)
        - hand_height: Max wrist Y position (normalized, inverted so higher is larger)
        """
        features = {
            'hand_to_mouth': 1.0, # Default: Far
            'hand_to_ear': 1.0,   # Default: Far
            'hand_height': 0.0    # Default: Low
        }
        
        if not pose_landmarks or not face_landmarks:
            return features
            
        # Keypoints
        # Pose: 15 (Left Wrist), 16 (Right Wrist)
        # Face: 13 (Upper Lip), 14 (Lower Lip) -> Mouth Center
        # Face: 234 (Left Ear), 454 (Right Ear)
        
        def get_pose_point(idx):
            lm = pose_landmarks.landmark[idx]
            # Check visibility
            if lm.visibility < 0.5: return None
            return np.array([lm.x * w, lm.y * h])
            
        def get_face_point(idx):
            lm = face_landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
            
        left_wrist = get_pose_point(15)
        right_wrist = get_pose_point(16)
        
        mouth_center = (get_face_point(13) + get_face_point(14)) / 2.0
        left_ear = get_face_point(234)
        right_ear = get_face_point(454)
        
        # Normalization factor (face height)
        face_top = get_face_point(10)
        face_bottom = get_face_point(152)
        face_height = np.linalg.norm(face_top - face_bottom)
        if face_height == 0: face_height = 1.0
        
        wrists = [w for w in [left_wrist, right_wrist] if w is not None]
        
        if not wrists:
            return features
            
        # 1. Hand to Mouth (Min distance)
        mouth_dists = [np.linalg.norm(wrist - mouth_center) for wrist in wrists]
        features['hand_to_mouth'] = min(mouth_dists) / face_height
        
        # 2. Hand to Ear (Min distance)
        # Check both ears for both wrists
        ear_dists = []
        for wrist in wrists:
            ear_dists.append(np.linalg.norm(wrist - left_ear))
            ear_dists.append(np.linalg.norm(wrist - right_ear))
        features['hand_to_ear'] = min(ear_dists) / face_height
        
        # 3. Hand Height (Max Y, inverted because Y is down)
        # We want "High" to be a large number.
        # Y=0 is top, Y=1 is bottom. So (1 - y) gives height from bottom.
        # Or just use raw Y relative to face center?
        # Let's use: (Chin Y - Wrist Y) / Face Height. Positive = Above chin.
        chin_y = face_bottom[1]
        heights = [(chin_y - wrist[1]) / face_height for wrist in wrists]
        features['hand_height'] = max(heights)
        
        return features

    def _extract_frown_features(self, face_landmarks, h, w):
        """
        Extracts features specific to frowning:
        - brow_eye_dist: Normalized distance between eyebrow and eye.
        - mouth_curvature: Vertical difference between mouth corners and center.
        """
        features = {
            'brow_eye_dist': 0.0,
            'mouth_curvature': 0.0
        }
        
        if not face_landmarks:
            return features
            
        def get_point(idx):
            lm = face_landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
            
        # 1. Brow-Eye Distance
        # Left Eyebrow Middle (105) -> Left Eye Top (159)
        # Right Eyebrow Middle (334) -> Right Eye Top (386)
        l_brow = get_point(105)
        l_eye = get_point(159)
        r_brow = get_point(334)
        r_eye = get_point(386)
        
        l_dist = np.linalg.norm(l_brow - l_eye)
        r_dist = np.linalg.norm(r_brow - r_eye)
        
        # Normalize by face height (approx)
        face_top = get_point(10)
        face_bottom = get_point(152)
        face_height = np.linalg.norm(face_top - face_bottom)
        if face_height == 0: face_height = 1.0
        
        features['brow_eye_dist'] = ((l_dist + r_dist) / 2.0) / face_height
        
        # 2. Mouth Curvature (Sadness)
        # Corners: 61 (Left), 291 (Right)
        # Center: Average of 0 (Top) and 17 (Bottom) is center, but we want the "line"
        # Let's compare Corner Y vs Center Y.
        # If Corners are LOWER than Center (higher Y value), it's a frown/sad mouth.
        # If Corners are HIGHER than Center (lower Y value), it's a smile.
        
        left_corner = get_point(61)
        right_corner = get_point(291)
        top_lip = get_point(0)
        bottom_lip = get_point(17)
        
        mouth_center_y = (top_lip[1] + bottom_lip[1]) / 2.0
        corners_y = (left_corner[1] + right_corner[1]) / 2.0
        
        # Positive = Corners are below center (Frown)
        # Negative = Corners are above center (Smile)
        features['mouth_curvature'] = (corners_y - mouth_center_y) / face_height
        
        return features

