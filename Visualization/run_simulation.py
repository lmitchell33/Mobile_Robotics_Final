import os, sys
os.add_dll_directory(r"C:\Users\Nate\Documents\gtsam\build\bin\Release")
sys.path.append(r"C:\Users\Nate\Documents\gtsam\build\python")

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gtsam

from Graph.visualizer_2d import DynamSLAMVisualizer2D
from factor.build_factor import BuildFactor


# ---------------- Utilities ----------------

def quaternion_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def make_pose(x, y, yaw):
    R = gtsam.Rot3.Yaw(yaw)
    t = gtsam.Point3(x, y, 0.0)
    return gtsam.Pose3(R, t)


def imu_preintegrate(acc_body, gyro_body, imu_params,
                     acc_var, gyro_var, dt_imu, steps):
    bias = gtsam.imuBias.ConstantBias()
    pim = gtsam.PreintegratedImuMeasurements(imu_params, bias)

    for _ in range(steps):
        a = acc_body + np.random.randn(3) * np.sqrt(acc_var)
        w = gyro_body + np.random.randn(3) * np.sqrt(gyro_var)
        pim.integrateMeasurement(a, w, dt_imu)

    return pim


# ---------------- Simulation ----------------

def run_sim():

# ---------------- Timing ----------------
    fps = 10.0
    dt_cam = 1.0 / fps

    imu_rate = 200.0
    dt_imu = 1.0 / imu_rate
    imu_steps = int(dt_cam / dt_imu)

    NUM_FRAMES = 300  

    # ---------------- Camera ----------------
    fx, fy, s, cx, cy = 220, 220, 0, 320, 240

    # ---------------- SLAM ----------------
    slam = BuildFactor(
        window_size=6,
        fps=fps,
        fx=fx, fy=fy, s=s, cx=cx, cy=cy
    )

    slam.initialize_first_pose(
        make_pose(0, 0, 0),
        vel0=np.array([0.4, 0.0, 0.0])
    )

    # ---------------- IMU Params ----------------
    imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    imu_params.n_gravity = np.array([0, 0, -9.81])

    acc_var = (0.03)**2
    gyro_var = (0.01)**2

    imu_params.setAccelerometerCovariance(np.eye(3) * acc_var)
    imu_params.setGyroscopeCovariance(np.eye(3) * gyro_var)
    imu_params.setIntegrationCovariance(np.eye(3) * 1e-6)

    # ---------------- World ----------------
    static_landmarks = np.array([
        [5,  2, 3],
        [6, -2, 3],
        [10, 3, 3],
        [15, -4, 3],
        [20, 4, 3],
        [25, -1, 3],
    ])

    dyn_init = np.array([
        [4, -2, 3.5],
        [8,  3, 3.5],
        [14, -3, 3.5],
    ])

    dyn_vel = np.array([
        [0.05,  0.01, 0],
        [-0.04, 0.02, 0],
        [0.03, -0.03, 0]
    ])

    poses = []
    dyn_per_frame = []

    # ---------------- Loop ----------------
    for k in range(1, NUM_FRAMES + 1):

        t = k * dt_cam

        # Smooth turn
        # --------- MOTION SCHEDULE ----------
# Straight → Turn → Straight

        if k < 40:
            # Phase 1: straight
            yaw_rate = 0.0

        elif 40 <= k < 90:
            # Phase 2: constant left turn
            yaw_rate = 0.15      # rad/s  (~8.6°/s)

        elif 90 <= k < 250:
            # Phase 3: start turning back
            yaw_rate = -0.15
        else:
            # Phase 3: start turning back
            yaw_rate = 0.05
        # forward speed (same as initial velocity)
        v = 0.4  # m/s

        # yaw rate (already defined)
        gyro = np.array([0.0, 0.0, yaw_rate])

        # centripetal acceleration: a_y = v * omega
        acc = np.array([
            0.0,            # no forward acceleration
            v * yaw_rate,   # lateral acceleration causes turn
            0.0
        ])
        pim = imu_preintegrate(
            acc, gyro,
            imu_params,
            acc_var, gyro_var,
            dt_imu, imu_steps
        )

        slam.add_imu_factor(pim, timestamp=t)

        pose = slam.get_pose(k)
        if pose is None:
            continue

        T_cw = pose.transformTo

        # -------- Static features --------
        obs = []
        for i, Pw in enumerate(static_landmarks):
            Pc = T_cw(gtsam.Point3(*Pw))
            if Pc[2] < 0.5:
                continue

            u = fx * Pc[0]/Pc[2] + cx
            v = fy * Pc[1]/Pc[2] + cy

            obs.append(type("Obs", (), {
                "feature_id": i,
                "pixel_coords": np.array([u, v]),
                "point_3d": np.array([Pc[0], Pc[1], Pc[2]]),
                "is_static": True
            }))

        slam.add_feature_observations(obs, timestamp=t)

        # -------- Dynamic features --------
        dyn_list = []
        for j in range(len(dyn_init)):
            Pw = dyn_init[j] + dyn_vel[j] * k

            Pc = T_cw(gtsam.Point3(*Pw))
            if Pc[2] < 0.5:
                continue

            u = fx * Pc[0]/Pc[2] + cx
            v = fy * Pc[1]/Pc[2] + cy

            slam.add_virtual_landmark(
                feature_id=100+j,
                predicted_world=Pw,
                uv=np.array([u, v]),
                timestamp=t,
            )

            dyn_list.append({
                "pos": [Pw[0], Pw[1]],
                "cov": [[0.15, 0], [0, 0.15]],
                "id": 100+j,
                "color": "red"
            })

        dyn_per_frame.append(dyn_list)

        est = slam.get_current_trajectory()[-1][1]
        yaw = quaternion_to_yaw(est[3:7])
        poses.append([est[0], est[1], yaw])

    poses = np.array(poses)

    static_map = np.array([
        [p[0], p[1]] for p in slam.get_landmarks().values()
    ])

    # ---------------- Visualize ----------------
    viz = DynamSLAMVisualizer2D(static_map=static_map)
    viz.animate(poses, dyn_per_frame)


if __name__ == "__main__":
    run_sim()

