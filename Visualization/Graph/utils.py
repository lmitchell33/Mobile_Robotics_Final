import numpy as np

# create a smooth motion of the camera view following the robot as it moves
# interpolate camera center
def smooth_follow(targetXY, previousCenter, alpha=0.1):
    if previousCenter is None:
        return np.array(targetXY, dtype=float)
    return previousCenter * (1-alpha) + np.array(targetXY) * alpha