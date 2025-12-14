import cv2
import numpy as np
import os
from dataclasses import dataclass
import gtsam

TEST_ROS_BAG = os.path.expanduser("~/Downloads/MH_01_easy.bag")

BASELINE_DISTANCE = 0.11008
CAMERA_MATRIX = np.array([
    [458.654, 0.0, 367.215], 
    [0.0, 457.296, 248.375], 
    [0.0, 0.0, 1.0]
])
CAMERA_MATRIX_1 = np.array([
    [457.587, 0.0, 379.999],
    [0.0, 456.134, 255.238],
    [0.0, 0.0, 1.0]
])

IMU_T_CAM = np.array([
    [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975], 
    [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768], 
    [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949], 
    [0.0, 0.0, 0.0, 1.0]
])

distortion_coeffs_cam0 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
distortion_coeffs_cam1 = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05])
resolution = (752, 480) 

R_cam0_cam1 = np.array([
    [ 0.99999726,  0.00230502,  0.00040276],
    [-0.00230153,  0.99999232, -0.00388445],
    [-0.0004114 ,  0.00388356,  0.99999242]
])
t_cam0_cam1 = np.array([0.11007391, -0.00039933, -0.00037642])

@dataclass
class FeatureObservation:
    feature_id: str
    pixel_coords: np.ndarray
    point_3d: np.ndarray
    is_static: bool = True
    uncertainty: np.ndarray = None

class ImageProcessor:
    def __init__(self, camera_matrix: np.ndarray = CAMERA_MATRIX, baseline_distance: float = BASELINE_DISTANCE, imu_t_matrix: np.ndarray = IMU_T_CAM):
        self._camera_matrix = camera_matrix
        self._baseline_distance = baseline_distance
        self._imu_t_matrix = imu_t_matrix

        # state values for tracking
        self._prev_left_image = None
        self._prev_features = None
        self._next_feature_id = 0
        self._prev_ids = None
        self._prev_points_3d = None

        # stereo camera object I can use to compute the disparity with. Paper says they use semiglobal block matching hence SGBM and not BM
        self._stereo_cam = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128, # must be divisible by 16
            blockSize=5,
            P1=8*3*5**2,
            P2=32*3*5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2
        )

        # NEW STUFF
        self._dist_left = distortion_coeffs_cam0
        self._dist_right = distortion_coeffs_cam1

        # we will use a rectified intrinsic for projection / 3D
        image_size = resolution  # (w, h)

        # Compute rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    CAMERA_MATRIX, self._dist_left,
    CAMERA_MATRIX_1, self._dist_right,
            image_size,
            R_cam0_cam1,
            t_cam0_cam1,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        # Rectification maps
        self._left_map1, self._left_map2 = cv2.initUndistortRectifyMap(
            self._camera_matrix, self._dist_left,
            R1, P1, image_size, cv2.CV_32FC1
        )
        self._right_map1, self._right_map2 = cv2.initUndistortRectifyMap(
            CAMERA_MATRIX_1, self._dist_right,
            R2, P2, image_size, cv2.CV_32FC1
        )

        # Use rectified K for projection/3D
        self._rect_K = P1[0:3, 0:3]
        
        fx = P1[0,0]
        fy = P1[1,1]
        cx = P1[0,2]
        cy = P1[1,2]

        self.K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    def extract_features(self, left_image: np.ndarray):
        """Detects new features in the current frame (left image) using the Harris corner detector as stated on page 293 of the paper"""
        # goodFeaturesToTrack only takes a grayscale image, so if there are RGB channels, convert to grayscale
        if len(left_image.shape) == 3:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        
        corners = cv2.goodFeaturesToTrack(
            image=left_image, 
            maxCorners=800, 
            qualityLevel=0.01, 
            minDistance=7, 
            blockSize=7, 
            useHarrisDetector=True
        )

        if corners is None:
            return np.empty((0, 2))
        return corners.reshape(-1, 2)

    def track_features(self, prev_image: np.ndarray, next_image: np.ndarray, prev_points: np.ndarray):
        """Track features using LK optical flow as stated on page 293 of the paper"""
        # TODO: update the other parameters here?
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        prev_points = prev_points.reshape(-1, 1, 2).astype(np.float32)
        new_positions, status, err = cv2.calcOpticalFlowPyrLK(prev_image, next_image, prev_points, None, **lk_params)

        if status is None:
            return np.array([]), np.array([]), np.array([], dtype=bool)

        # only take the good points that were successfully tracker
        status = status.reshape(-1).astype(bool)
        good_prev = prev_points[status.flatten() == 1].reshape(-1, 2)
        good_curr = new_positions[status.flatten() == 1].reshape(-1, 2)

        return good_prev, good_curr, status

    def calculate_disparity(self, left_image: np.ndarray, right_image: np.ndarray):
        # look at opencv_source_code/samples/python/stereo_match.py for more examples
        return self._stereo_cam.compute(left_image, right_image).astype(np.float32) / 16.0

    def calculate_3D_location(self, x: int, y: int, disparity: float):
        """
        Uses equations 6 and 7 from the paper (page 293) to calculate the 3D locations of the features/landmarks. The following formulas are used: 

        c = [cx; cy] is the principal point relative to the physical imaging plane
        z = coefficient = (fx * baseline_distance) / disparity
        x = coefficient * (x - cx) / fx
        y = coefficient * (y - cy) / fy
        """
        cx, cy = self._rect_K[0, 2], self._rect_K[1, 2]
        fx, fy = self._rect_K[0, 0], self._rect_K[1, 1]

        # divide by 0 zeros
        if disparity <= 0:
            return None

        Z = (fx * self._baseline_distance) / disparity
        X = Z * (x - cx) / fx
        Y = Z * (y - cy) / fy
        return np.array([X, Y, Z])

    def calculate_scene_flow(self, prev_3d_points, curr_3d_points, imu_pose_delta):
        """
        Uses equation 8 to find the scene flow:

        M = {Pj - R(q)Pi 0 alpha}
        """
        if imu_pose_delta is None:
            return np.array([])

        # NOTE: might have to add the imu_pose_delta param to the process_frame function
        transform = self._imu_t_matrix @ imu_pose_delta @ np.linalg.inv(self._imu_t_matrix)
        R = transform[:3, :3]
        p = transform[:3, 3]

        scene_flows = []
        for Pi, Pj in zip(prev_3d_points, curr_3d_points):
            Pj_pred = R @ Pi + p
            delta_M = Pj - Pj_pred

            # try to add something with an uncertainy model and the Mahalanobis distance if there is time

            scene_flows.append(delta_M)

        return scene_flows

    def process_frame(self, left_image: np.ndarray, right_image: np.ndarray, imu_pose_delta=None):
        """Processes an entire frame/image"""
        if len(left_image.shape) == 3:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)

        if len(right_image.shape) == 3:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        left_image = cv2.remap(left_image, self._left_map1, self._left_map2, cv2.INTER_LINEAR)
        right_image = cv2.remap(right_image, self._right_map1, self._right_map2, cv2.INTER_LINEAR)

        # feature detection detection & tracking
        if self._prev_left_image is None:
            # first frame
            curr_features = self.extract_features(left_image)
            num_features = curr_features.shape[0]
            curr_ids = np.arange(num_features, dtype=int)
            self._next_feature_id += num_features
        else:
            # not first frame: track progression of features and bookkeep
            prev_features = self._prev_features
            prev_ids = self._prev_ids
            good_prev, good_curr, status_mask = self.track_features(
                self._prev_left_image, left_image, prev_features
            )

            # apply mask to keep only successfully tracked points
            curr_ids = prev_ids[status_mask]
            curr_features = good_curr

        # disparity
        disparity = self.calculate_disparity(left_image, right_image)

        # 3D points in IMU frame
        curr_3d_points = []
        for fid, point in zip(curr_ids, curr_features):
            x, y = int(point[0]), int(point[1])

            if x < 0 or x >= disparity.shape[1] or y < 0 or y >= disparity.shape[0]:
                curr_3d_points.append(None)
                continue

            disp_val = disparity[y, x]
            point_3d = self.calculate_3D_location(x, y, disp_val)
            if point_3d is None:
                curr_3d_points.append(None)
                continue

            P_cam = np.append(point_3d, 1.0)
            P_imu = np.linalg.inv(IMU_T_CAM) @ P_cam
            point_3d_imu = P_imu[:3]

            curr_3d_points.append(point_3d_imu)

        # Scene flow estimation and dynamic classification
        scene_flows = None
        dynamic_ids = set()

        if self._prev_points_3d is not None and imu_pose_delta is not None:
            previous_id_to_index = {fid: i for i, fid in enumerate(self._prev_ids)}

            matched_prev_3d = []
            matched_curr_3d = []
            matched_ids = []

            for fid, curr_point in zip(curr_ids, curr_3d_points):
                if fid in previous_id_to_index and curr_point is not None:
                    prev_index = previous_id_to_index[fid]
                    previous_point = self._prev_points_3d[prev_index]

                    if previous_point is None:
                        continue

                    matched_prev_3d.append(previous_point)
                    matched_curr_3d.append(curr_point)
                    matched_ids.append(fid)

            if len(matched_prev_3d) > 0:
                scene_flows = self.calculate_scene_flow(
                    np.array(matched_prev_3d),
                    np.array(matched_curr_3d),
                    imu_pose_delta
                )

                # classify dynamic vs static by flow magnitude
                motion_threshold = 0.50  # tune this
                for fid, flow in zip(matched_ids, scene_flows):
                    if np.linalg.norm(flow) > motion_threshold:
                        dynamic_ids.add(fid)

        # create the wrapper class
        observations = []
        for fid, point, point_3d in zip(curr_ids, curr_features, curr_3d_points):
            if point_3d is None:
                continue

            x, y = int(point[0]), int(point[1])

            observation = FeatureObservation(
                feature_id=fid,
                pixel_coords=np.array([x, y]),
                point_3d=point_3d,
                is_static=(fid not in dynamic_ids),
                uncertainty=None
            )
            observations.append(observation)

        # bookkeaping
        self._prev_left_image = left_image
        self._prev_features = curr_features
        self._prev_ids = curr_ids
        self._prev_points_3d = np.array(curr_3d_points, dtype=object)

        return observations, scene_flows


if __name__ == "__main__":
    from data_loader import DataLoader

    image_processor = ImageProcessor()
    data_loader = DataLoader(TEST_ROS_BAG)
    for timestamp, left, right in data_loader.get_stereo():
        image_processor.process_frame(left, right, imu_pose_delta = np.eye(4))
        break