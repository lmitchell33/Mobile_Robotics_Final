import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from .uncertainty import draw_uncertainty_ellipse
from .utils import smooth_follow

class DynamSLAMVisualizer2D:
    def __init__(self, static_map=None, gt_path_xy=None, trail_length=250):
        self.static_map = static_map     
        self.gt_xy = gt_path_xy
        self.trail_length = trail_length

        self.poses = None
        self.dynamic_obs = None
        self.view_center = None

        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        self.robot_dot,   = self.ax.plot([], [], 'bo', markersize=8, label="Robot")
        self.robot_trail, = self.ax.plot([], [], 'b-', linewidth=2)

        self.wall_artists = []
        self.dynamic_artists = None
        self.gt_plot = None
        self.fov_patch = None

    def _draw_fov(self, x, y, theta, fov_deg=90, depth=6):
        half = np.radians(fov_deg / 2)

        left_angle = theta + half
        right_angle = theta - half

        left_pt = (x + depth * np.cos(left_angle),
                   y + depth * np.sin(left_angle))
        right_pt = (x + depth * np.cos(right_angle),
                    y + depth * np.sin(right_angle))

        verts = np.array([
            [x, y],
            left_pt,
            right_pt
        ])

        return Polygon(
            verts,
            closed=True,
            facecolor='cyan',
            alpha=0.18,
            edgecolor='cyan',
            linewidth=1.2
        )

    def _init_plot(self):
        self.ax.set_title("Dynam-SLAM Visualization (2D)")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)

        # ---- Draw static wall segments ----
        if self.static_map is not None:
            for seg in self.static_map:
                p1, p2 = seg
                xs = [p1[0], p2[0]]
                ys = [p1[1], p2[1]]

                line, = self.ax.plot(
                    xs, ys,
                    color="gray",
                    linewidth=4,
                    alpha=0.9,
                    label="Wall" if len(self.wall_artists) == 0 else ""
                )
                self.wall_artists.append(line)

        # Ground truth path
        if self.gt_xy is not None:
            self.gt_plot, = self.ax.plot(
                self.gt_xy[:, 0], self.gt_xy[:, 1],
                'g--', linewidth=1.5, label="Ground Truth"
            )

        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.legend()
        return []

    def _update(self, frame):
        if frame >= len(self.poses):
            return []
        x, y, theta = self.poses[frame]

        # camera follow
        self.view_center = smooth_follow((x, y), self.view_center)
        cx, cy = self.view_center
        self.ax.set_xlim(cx - 10, cx + 10)
        self.ax.set_ylim(cy - 10, cy + 10)

        # Robot dot
        self.robot_dot.set_data([x], [y])

        # Robot trail
        start = max(0, frame - self.trail_length)
        traj = self.poses[start:frame + 1]
        self.robot_trail.set_data(traj[:, 0], traj[:, 1])

        # Dynamic obstacles
        dyn = self.dynamic_obs[frame]

        xs = [obs["pos"][0] for obs in dyn]
        ys = [obs["pos"][1] for obs in dyn]

        if self.dynamic_artists is None:
            self.dynamic_artists = self.ax.scatter(xs, ys, c="red", s=30, label="Dynamic Obstacles")
        else:
            self.dynamic_artists.set_offsets(np.column_stack([xs, ys]))

        # Clear old uncertainty + FOV
        for p in list(self.ax.patches):
            p.remove()

        # Draw uncertainty ellipses
        for obs in dyn:
            pos = obs["pos"]
            cov = obs["cov"]

            draw_uncertainty_ellipse(
                self.ax,
                pos,
                cov=cov,
                n_std=2,
                edgecolor="red",
                linewidth=1,
                alpha=0.7
            )

        # Draw FOV
        self.fov_patch = self._draw_fov(x, y, theta)
        self.ax.add_patch(self.fov_patch)

        return []

    def animate(self, poses, dynamic_obstacles, interval=40):
        self.poses = poses
        self.dynamic_obs = dynamic_obstacles

        anim = FuncAnimation(
            self.fig,
            self._update,
            frames=len(poses),
            init_func=self._init_plot,
            interval=interval,
            blit=False,
            repeat=False
        )
        plt.show()
        return anim
