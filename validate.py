import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_groundtruth(gt_path):
    data = np.loadtxt(gt_path)

    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quats = data[:, 4:8]

    # they are quaternions 
    rots = R.from_quat(quats)
    yaws = rots.as_euler("zyx")[:, 0]
    return timestamps, positions, yaws

# from claude. hoping this could explain the results
def interpolate_groundtruth(gt_times, gt_pos, gt_yaw, slam_times):
    """
    Interpolate ground truth trajectory to match SLAM timestamps
    """
    interp_x = np.interp(slam_times, gt_times, gt_pos[:, 0])
    interp_y = np.interp(slam_times, gt_times, gt_pos[:, 1])
    interp_yaw = np.interp(slam_times, gt_times, gt_yaw)

    return np.vstack([interp_x, interp_y, interp_yaw]).T

def compute_ate(slam_traj, gt_traj):
    pos_error = slam_traj[:, :2] - gt_traj[:, :2]
    ate_rmse = np.sqrt(np.mean(np.sum(pos_error**2, axis=1)))
    return ate_rmse, pos_error

def validate_trajectory(slam_times, slam_traj, gt_path):
    gt_times, gt_pos, gt_yaw = load_groundtruth(gt_path)
    gt_times = gt_times - gt_times[0] + slam_times[0]
    gt_interp = interpolate_groundtruth(gt_times, gt_pos, gt_yaw, slam_times)
    ate_rmse, pos_error = compute_ate(slam_traj, gt_interp)

    plt.plot(np.linalg.norm(pos_error, axis=1))
    plt.title("Position Error Over Time (ATE RMSE)")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.grid()
    plt.show()

    return ate_rmse