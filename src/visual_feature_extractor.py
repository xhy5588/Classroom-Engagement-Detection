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
        
        # 4. Body Pose Features (Hand Raise)
        is_hand_raised = self._detect_handraise(pose_landmarks, h, w)
        
        # 5. Hand-to-Face Distance
        hand_face_dist = self._calculate_hand_face_distance(pose_landmarks, face_landmarks, h, w)

        # 6. Frowning Detection
        frown_score = self._detect_frown(face_landmarks, h, w)

        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'ear_left': ear_left,
            'ear_right': ear_right,
            'ear': avg_ear,
            'mar': mar,
            'hand_face_dist': hand_face_dist,
            'hand_raised': is_hand_raised,
            'frown_score': frown_score
        }
        
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
        return mar

    def _calculate_hand_face_distance(self, pose_landmarks, face_landmarks, h, w):
        """
        Calculate the minimum distance between wrists and the mouth/ears.
        Returns normalized distance (relative to image diagonal or similar).
        """
        if not pose_landmarks or not face_landmarks:
            return 1.0 # Max distance if not detected

        # Pose Indices: 15 (Left Wrist), 16 (Right Wrist)
        # Face Indices: 0 (Lips Center), 234 (Left Ear), 454 (Right Ear)
        
        def get_pose_point(idx):
            lm = pose_landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
            
        def get_face_point(idx):
            lm = face_landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])

        left_wrist = get_pose_point(15)
        right_wrist = get_pose_point(16)
        
        mouth = get_face_point(0)
        left_ear = get_face_point(234)
        right_ear = get_face_point(454)
        
        # Calculate distances
        # We care about ANY hand being close to ANY relevant face part
        targets = [mouth, left_ear, right_ear]
        wrists = [left_wrist, right_wrist]
        
        min_dist = float('inf')
        
        for wrist in wrists:
            for target in targets:
                dist = np.linalg.norm(wrist - target)
                if dist < min_dist:
                    min_dist = dist
                    
        # Normalize by face width (approx) to be scale invariant
        # Face width ~ distance between ears (234 and 454)
        face_width = np.linalg.norm(left_ear - right_ear)
        if face_width > 0:
            normalized_dist = min_dist / face_width
        else:
            normalized_dist = min_dist / 100.0 # Fallback
            
        return normalized_dist

    def _detect_frown(self, landmarks, h, w):
        """
        Detects frowning based on the distance between inner eyebrows.
        Returns a score (lower distance = more likely frowning).
        """
        # Indices for inner eyebrows: 107 (Left), 336 (Right)
        # Indices for outer eyes (for normalization): 33 (Left), 263 (Right)
        
        def get_point(idx):
            lm = landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
            
        inner_brow_left = get_point(107)
        inner_brow_right = get_point(336)
        
        outer_eye_left = get_point(33)
        outer_eye_right = get_point(263)
        
        brow_dist = np.linalg.norm(inner_brow_left - inner_brow_right)
        eye_dist = np.linalg.norm(outer_eye_left - outer_eye_right)
        
        if eye_dist == 0:
            return 0.0
            
        # Normalized brow distance
        norm_brow_dist = brow_dist / eye_dist
        
        return norm_brow_dist

    def _detect_handraise(self, pose_landmarks, h, w):
        """
        Detects if either hand is raised above the shoulders.
        """
        if not pose_landmarks:
            return False
            
        # Indices: 
        # 11: Left Shoulder, 12: Right Shoulder
        # 15: Left Wrist, 16: Right Wrist
        
        left_shoulder_y = pose_landmarks.landmark[11].y
        right_shoulder_y = pose_landmarks.landmark[12].y
        
        left_wrist_y = pose_landmarks.landmark[15].y
        right_wrist_y = pose_landmarks.landmark[16].y
        
        # Y increases downwards. So higher (in space) means lower Y value.
        # Check if wrist is significantly above shoulder (e.g. 5% of height buffer)
        
        threshold = 0.05 
        
        left_raised = left_wrist_y < (left_shoulder_y - threshold)
        right_raised = right_wrist_y < (right_shoulder_y - threshold)
        
        return left_raised or right_raised

