import os, sys
os.add_dll_directory(r"C:\Users\Nate\Documents\gtsam\build\bin\Release")
sys.path.append(r"C:\Users\Nate\Documents\gtsam\build\python")


import numpy as np
import gtsam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from build_factor import BuildFactor  # change if needed

'''
AI generated test script for the factor graph builder
'''

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def make_pose(x, y, z):
    R = gtsam.Rot3()
    t = gtsam.Point3(x, y, z)
    return gtsam.Pose3(R, t)


def fake_preintegrated_imu(dt):
    imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    imu_params.setAccelerometerCovariance(np.eye(3)*1e-4)
    imu_params.setGyroscopeCovariance(np.eye(3)*1e-4)
    imu_params.setIntegrationCovariance(np.eye(3)*1e-4)

    pim = gtsam.PreintegratedImuMeasurements(
        imu_params,
        gtsam.imuBias.ConstantBias()
    )

    acc = np.array([0.1, 0.0, 0.0])  # forward acceleration
    omega = np.zeros(3)

    pim.integrateMeasurement(acc, omega, dt)
    return pim


class FakeObs:
    def __init__(self, fid, uv, point3d, is_static=True):
        self.feature_id = fid
        self.pixel_coords = uv
        self.point_3d = point3d
        self.is_static = is_static
        self.uncertainty = None


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def visualize(static_lm, traj, dynamic_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # trajectory
    traj_points = np.array([vec for _, vec in traj])
    ax.plot(traj_points[:,0], traj_points[:,1], traj_points[:,2], 'b-', label="Camera path")

    # static LM
    if len(static_lm) > 0:
        stat = np.array(list(static_lm.values()))
        ax.scatter(stat[:,0], stat[:,1], stat[:,2], c='green', s=40, label="Static LM")

    # dynamic LM
    if len(dynamic_points) > 0:
        dyn = np.vstack(dynamic_points)
        ax.scatter(dyn[:,0], dyn[:,1], dyn[:,2], c='red', s=20, label="Dynamic LM")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


# ---------------------------------------------------------
# Main Test
# ---------------------------------------------------------
def run_test():
    fps = 10
    dt = 1.0 / fps

    # camera intrinsics
    fx, fy, s, cx, cy = 200, 200, 0, 320, 240

    slam = BuildFactor(window_size=4, fps=fps, fx=fx, fy=fy, s=s, cx=cx, cy=cy)
    slam.initialize_first_pose(make_pose(0,0,0), vel0=np.zeros(3))

    static_world = np.array([[1.0,  0.0, 2.0],
                             [1.5, -0.5, 3.0],
                             [2.0,  1.0, 2.5]])

    # dynamic features: 5 points, initial positions + velocity
    dyn_initial = np.array([
        [0.0, 0.0, 4.0],
        [-1.0, 0.5, 3.0],
        [0.5, -1.0, 3.5],
        [1.0, 1.0, 4.0],
        [-0.5, -0.5, 3.2]
    ])
    dyn_vel = np.array([
        [0.1, 0.05, 0.0],
        [0.15, 0.0, 0.0],
        [0.07, -0.03, 0.0],
        [0.08, 0.02, 0.0],
        [0.12, -0.04, 0.0]
    ])

    dynamic_positions_all_frames = []  # store for visualization

    # simulate 10 frames
    for frame in range(1, 11):
        print(f"\n--- Frame {frame} ---")

        # IMU integration
        pim = fake_preintegrated_imu(dt)
        slam.add_imu_factor(pim, timestamp=frame*dt)

        pose = slam.get_pose(frame)
        tform = pose.transformTo

        # STATIC LANDMARKS
        static_obs_list = []
        for i, p in enumerate(static_world):
            pc = tform(gtsam.Point3(*p))
            x, y, z = pc[0], pc[1], pc[2]
            uv = np.array([fx*x/z + cx, fy*y/z + cy])
            static_obs_list.append(FakeObs(i, uv, np.array([x,y,z]), True))

        slam.add_feature_observations(static_obs_list, timestamp=frame*dt)

        # DYNAMIC LANDMARKS
        dynamic_points_this_frame = []

        for k in range(len(dyn_initial)):
            pos_world = dyn_initial[k] + dyn_vel[k] * frame

            pc = tform(gtsam.Point3(*pos_world))
            x, y, z = pc[0], pc[1], pc[2]
            uv = np.array([fx*x/z + cx, fy*y/z + cy])

            dynamic_points_this_frame.append(pos_world)

            slam.add_virtual_landmark(100 + k, pos_world, uv, timestamp=frame*dt)

        dynamic_positions_all_frames.extend(dynamic_points_this_frame)

    # results
    traj = slam.get_current_trajectory()
    static_lm = slam.get_landmarks()

    print("\nTrajectory:")
    print(traj)

    print("\nStatic landmarks:")
    print(static_lm)

    print("\nDynamic stored:", len(dynamic_positions_all_frames))

    visualize(static_lm, traj, dynamic_positions_all_frames)


if __name__ == "__main__":
    run_test()
