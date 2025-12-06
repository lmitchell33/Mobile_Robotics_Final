from Graph.visualizer_2d import DynamSLAMVisualizer2D
from Graph.dummy_data import generate_dummy_data

if __name__ == "__main__":
    poses, static_map, dynamic_obs = generate_dummy_data()
    viz = DynamSLAMVisualizer2D(static_map=static_map)
    viz.animate(poses, dynamic_obs)
