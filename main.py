import os
import numpy as np
import gtsam
from preprocess.imu import imu_preintegration_factory, IMUPreintegrator
from preprocess.image_processor import ImageProcessor, CAMERA_MATRIX
from preprocess.data_loader import DataLoader
from factor.build_factor import BuildFactor
from Visualization.Graph.visualizer_2d import DynamSLAMVisualizer2D
from validate import validate_trajectory

def imu_delta_transformation(imu_preintegration: IMUPreintegrator, prev_pose, prev_vel, prev_bias):
    pim = imu_preintegration.preintegration
    prev_state = gtsam.NavState(prev_pose, prev_vel)
    next_state = pim.predict(prev_state, prev_bias)
    return next_state.pose(), next_state.velocity()

def make_pose(x, y, z):
    R = gtsam.Rot3()
    t = gtsam.Point3(x, y, z)
    return gtsam.Pose3(R, t)

def run_dynam_slam(file_path):
    data = DataLoader(file_path)
    image_processor = ImageProcessor()
    imu = imu_preintegration_factory(gtsam.imuBias.ConstantBias())

    fx, fy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1]
    cx, cy = CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]
    s = CAMERA_MATRIX[0, 1]

    slam = BuildFactor(window_size=6, fps=20, fx=fx, fy=fy, s=s, cx=cx, cy=cy)
    slam.initialize_first_pose(make_pose(0,0,0), vel0=np.zeros(3))

    imu_dataset = data.get_imu()
    image_dataset = data.get_stereo()

    imu_timestep, imu_data = next(imu_dataset)
    image_timestamp, left_image, right_image = next(image_dataset)
    prev_imu_timestep = imu_timestep 

    trajectories = []
    dynamic_obs_frames = []
    timestamps = []
    frame_idx = 0

    for image_timestamp, left_image, right_image in image_dataset:

        while imu_timestep < image_timestamp:
            accel = np.array([
                imu_data.linear_acceleration.x,
                imu_data.linear_acceleration.y,
                imu_data.linear_acceleration.z
            ], dtype=float)

            gyro = np.array([
                imu_data.angular_velocity.x,
                imu_data.angular_velocity.y,
                imu_data.angular_velocity.z
            ], dtype=float)

            dt = float(imu_timestep - prev_imu_timestep)
            if dt > 0:
                imu.integrate_measurement(accel, gyro, dt)

            prev_imu_timestep = imu_timestep

            try:
                imu_timestep, imu_data = next(imu_dataset)
            except StopIteration:
                break

        slam.add_imu_factor(imu, image_timestamp)

        # reset IMU preintegration for next frame
        imu.reset()
        prev_imu_timestep = imu_timestep

        observations, scene_flows = image_processor.process_frame(left_image, right_image, np.eye(4))
        slam.add_feature_observations(observations, image_timestamp)

        pose, vel, bias = slam.get_latest_state()
        t = pose.translation()
        yaw = pose.rotation().yaw()

        trajectories.append([t[0], t[1], yaw])
        timestamps.append(image_timestamp)

        dynamic_list = []
        for obs in observations:
            if not obs.is_static and obs.point_3d is not None:
                Pc = gtsam.Point3(
                    float(obs.point_3d[0]),
                    float(obs.point_3d[1]),
                    float(obs.point_3d[2])
                )

                Pw = pose.transformFrom(Pc)
                pw_x, pw_y = Pw[0], Pw[1]

                cov = np.eye(2) * 0.1 
                dynamic_list.append({
                    "pos": np.array([pw_x, pw_y]),
                    "cov": cov
                })

        dynamic_obs_frames.append(dynamic_list)

    trajectories = np.asarray(trajectories, dtype=float)
    return trajectories, dynamic_obs_frames, timestamps

if __name__ == "__main__":
    data_file = os.path.expanduser("~/Downloads/MH_01_easy.bag")
    poses, dynamic_obs, timestamps = run_dynam_slam(data_file)
    # print(dynamic_obs)

    vis = DynamSLAMVisualizer2D()
    vis.animate(poses, dynamic_obs, interval=200)

    ground_truth = os.path.expanduser("~/Downloads/MH_01_easy.txt")
    validate_trajectory(timestamps, poses, ground_truth)