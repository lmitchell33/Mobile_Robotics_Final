import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Graph.visualizer_2d import DynamSLAMVisualizer2D
from factor.build_factor import BuildFactor 
import numpy as np

def quaternion_to_yaw(q):
    w, x, y, z = q
    # yaw from quaternion
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw

if __name__ == "__main__":
    # SLAM outputs from build_factor
    # bf = BuildFactor(
    #     window_size=5,
    #     fps=10,
    #     fx=500, fy=500, s=0,
    #     cx=320, cy=240
    # )

    # insert IMU + feature data here
    # DEMO ONLY: do nothing, just create fake output
    # Pretend BuildFactor produced some trajectory:
    trajectory = [
        (i, np.array([0.1*i, 0.05*i, 0, 1,0,0,0]))   # x,y,z,qw,qx,qy,qz
        for i in range(100)
    ]

    # Pretend it produced landmarks:
    landmarks = {
        1: np.array([2, 3, 0]),
        2: np.array([-1, 4, 0])
    }

    # Convert BuildFactor pose format to visualizer (x,y,theta)
    poses_xyth = []
    for idx, vec in trajectory:
        x, y, z = vec[0:3]
        q = vec[3:7]
        yaw = quaternion_to_yaw(q)
        poses_xyth.append([x, y, yaw])

    poses_xyth = np.array(poses_xyth)

    # Convert landmarks to array of (x, y)
    static_map = []
    for fid, p in landmarks.items():
        static_map.append([p[0], p[1]])

    static_map = np.array(static_map)

    # No dynamic obstacles yet, simple empty list per frame
    dynamic_obs = [[] for _ in range(len(poses_xyth))]

    # Run animation with BuildFactor outputs  
    viz = DynamSLAMVisualizer2D(static_map=static_map)
    viz.animate(poses_xyth, dynamic_obs)
