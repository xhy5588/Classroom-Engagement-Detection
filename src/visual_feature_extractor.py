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

        # 5. Hand Features (Pose)
        hand_raised = 0.0
        hand_near_mouth = 0.0
        
        if pose_landmarks:
            # Indices: 13(L_Elbow), 14(R_Elbow), 15(L_Wrist), 16(R_Wrist)
            l_elbow = pose_landmarks.landmark[13]
            r_elbow = pose_landmarks.landmark[14]
            l_wrist = pose_landmarks.landmark[15]
            r_wrist = pose_landmarks.landmark[16]
            
            # Check Hand Raised (Wrist above Elbow - y is inverted)
            if l_wrist.y < l_elbow.y or r_wrist.y < r_elbow.y:
                hand_raised = 1.0
                
            # Check Hand Near Mouth (Drinking/Yawning/Phone)
            mouth_center = np.mean([
                [face_landmarks.landmark[13].x, face_landmarks.landmark[13].y], # Upper lip
                [face_landmarks.landmark[14].x, face_landmarks.landmark[14].y]  # Lower lip
            ], axis=0)
            
            # Check multiple hand points: Wrist, Index Tip, Pinky Tip
            # Indices: L_Wrist(15), R_Wrist(16), L_Pinky(17), R_Pinky(18), L_Index(19), R_Index(20)
            
            def is_near(p_idx):
                p = pose_landmarks.landmark[p_idx]
                dist = np.linalg.norm([p.x - mouth_center[0], p.y - mouth_center[1]])
                return dist < 0.3 # Relaxed threshold (was 0.25)

            if (is_near(15) or is_near(17) or is_near(19) or  # Left Hand
                is_near(16) or is_near(18) or is_near(20)):   # Right Hand
                hand_near_mouth = 1.0

        # 6. Eyebrow Features (for Frowning)
        # Simple metric: Distance between eyebrow and eye
        # Left Eyebrow Middle (105) to Left Eye Top (159)
        # Right Eyebrow Middle (334) to Right Eye Top (386)
        # Normalize by inter-ocular distance or something stable? 
        # Let's normalize by face bounding box height (approx).
        
        brow_dist = 0.0
        try:
            l_brow = np.array([face_landmarks.landmark[65].x, face_landmarks.landmark[65].y]) # 105 is middle? 65 is closer to inner?
            # Let's use standard points. 
            # Left Brow: 70, 63, 105, 66, 107
            # Right Brow: 336, 296, 334, 293, 300
            # Left Eye Top: 159
            # Right Eye Top: 386
            
            l_brow_y = face_landmarks.landmark[105].y
            l_eye_y = face_landmarks.landmark[159].y
            r_brow_y = face_landmarks.landmark[334].y
            r_eye_y = face_landmarks.landmark[386].y
            
            # Distance is (Eye Y - Brow Y) because Y increases downwards. 
            # So Eye (lower) - Brow (higher) should be positive.
            # Frowning -> Brow goes down -> Distance decreases.
            
            l_dist_brow = l_eye_y - l_brow_y
            r_dist_brow = r_eye_y - r_brow_y
            
            brow_dist = (l_dist_brow + r_dist_brow) / 2.0
        except:
            brow_dist = 0.05 # Default

        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'ear_left': ear_left,
            'ear_right': ear_right,
            'ear': avg_ear,
            'mar': mar,
            'gaze_h': gaze_h,
            'gaze_v': gaze_v,
            'hand_raised': hand_raised,
            'hand_near_mouth': hand_near_mouth,
            'brow_dist': brow_dist
        }
        
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
        return mar

