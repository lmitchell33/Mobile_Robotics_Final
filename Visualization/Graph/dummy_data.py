import numpy as np

def generate_dummy_data(T=400, num_dyn=4):
    # robot path
    poses = []
    for t in range(T):
        x = 0.05 * t - 10
        y = 2.5 * np.sin(0.02 * t)
        theta = np.arctan2(
            2.5 * 0.02 * np.cos(0.02 * t),
            0.05
        )
        poses.append([x, y, theta])
    poses = np.array(poses)

    static_walls = []
    for i in range(-20, 21):
        static_walls.append([[i, -3], [i + 1, -3]])
        static_walls.append([[i, 3], [i + 1, 3]])

    static_walls = np.array(static_walls, dtype=float)

    # dynamic obstacles with uncertainty
    dynamicObs = []
    for t in range(T):
        obs_list = []
        for j in range(num_dyn):
            ox = -5 + 0.03 * t + np.sin(0.1 * j + t * 0.05)
            oy = np.cos(0.2 * j + t * 0.03)

            cov = np.array([
                [0.15 + 0.05*np.abs(np.sin(t/50)), 0.02],
                [0.02, 0.10 + 0.03*np.abs(np.cos(t/60))]
            ])

            obs_list.append({
                "pos": np.array([ox, oy]),
                "cov": cov
            })
        dynamicObs.append(obs_list)

    return poses, static_walls, dynamicObs
