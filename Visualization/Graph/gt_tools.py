import numpy as np

# minimum ground truth loader for trajectories
def load_ground_truth(path):
    data = np.loadtxt(path, delimiter= ",")
    return data[:, 1:3]