import cv2
import numpy as np
import os
from dataclasses import dataclass

TEST_ROS_BAG = os.path.expanduser("~/Downloads/MH_01_easy.bag")
# TODO: find the camera matrix, baseline distance, and imu transofrmation matrix for the dataset

@dataclass
class FeatureObservation:
    feature_id: str
    pixel_coords: np.ndarray
    point_3d: np.ndarray
    is_static: bool = True
    uncertainty: np.ndarray = None

class ImageProcessor:
    def __init__(self, camera_matrix: np.ndarray, baseline_distance: float, imu_t_matrix: np.ndarray):
        self._camera_matrix = camera_matrix
        self._baseline_distance = baseline_distance
        self._imu_t_matrix = imu_t_matrix

        # state values for tracking
        self._prev_left_image = None
        self._prev_features = None
        self._next_feature_id = 0
        self._prev_ids = None

        # stereo camera object I can use to compute the disparity with. Paper says they use semiglobal block matching hence SGBM and not BM
        self._stereo_cam = cv2.StereoSGBM__create(
            minDisparity=0,
            numDisparities=16*5, # must be divisible by 16
            blockSize=7,
        )

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
        new_positions, status, err = cv2.calcOpticalFlowPyrLK(prev_image, next_image, prev_points)

        # only take the good points that were successfully tracked
        status = status.reshape(-1).astype(bool)
        good_prev = prev_points[status.flatten() == 1]
        good_curr = new_positions[status.flatten() == 1]

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
        cx, cy = self._camera_matrix[0, 2], self._camera_matrix[1, 2]
        fx, fy = self._camera_matrix[0, 0], self._camera_matrix[1, 1]

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
        transform = self._imu_t_matrix @ imu_pose_delta @ np.linalg.inv(self._imu_t_matrix)
        R = transform[:3, :3]
        p = transform[3, :3]
        scene_flows = []

        for Pi, Pj in zip(prev_3d_points, curr_3d_points):
            Pj_pred = R @ Pi + p
            delta_M = Pj - Pj_pred

            # something with an uncertainy model and the Mahalanobis distance??

            scene_flows.append(delta_M)

        return scene_flows

    def process_frame(self, left_image: np.ndarray, right_image: np.ndarray):
        """Processes an entire frame/image"""
        if len(left_image.shape) == 3:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)

        if len(right_image.shape) == 3:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # feature detection & tracking
        if self._prev_left_image is None:
            # first frame, assign each feature a number 0-n
            features = self.extract_features(left_image)
            num_features = features.shape[0]
            curr_ids = range(0, num_features)
            self._next_feature_id += num_features

        else:
            # not first frame, track progression of features and bookeep
            prev_features = self._prev_features
            prev_ids = self._prev_ids
            good_prev, good_curr, status_mask = self.track_features(self._prev_left_image, left_image, prev_features)
            
            # apply a bitmap-esk operation to remove all feature ids that were not successfully tracked
            # status returns a list of bools indicating the success of the tracking of each point
            curr_ids = prev_ids[status_mask]
            curr_features = good_curr

        # calculate the disparity
        disparity = self.calculate_disparity(left_image, right_image)
        
        # get the 3D location of each feature/landmark
        observations = []
        for id, point in zip(curr_ids, features):
            x, y = int(point[0]), int(point[1])
            disp_val = disparity[y, x]
            point_3d = self.calculate_3D_location(x, y, disp_val)
            
            # TODO: implement uncertainty
            observation = FeatureObservation(
                feature_id=id,
                pixel_coords=np.array([x, y]),
                point_3d=point_3d,
                is_static=True,
                uncertainty=None
            )
            observations.append(observation)

        # bookeep for the next state
        self._prev_left_image = left_image
        self._prev_features = curr_features
        self._prev_ids = curr_ids

        return observations

if __name__ == "__main__":
    from data_loader import DataLoader
    data_loader = DataLoader(TEST_ROS_BAG)