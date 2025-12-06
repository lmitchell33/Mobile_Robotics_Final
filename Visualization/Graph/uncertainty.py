import numpy as np
from matplotlib.patches import Ellipse

# create an uncertainty ellispse for the dynamic obstacles in the environment
def draw_uncertainty_ellipse(ax, center, cov, n_std=2, **kwargs):

    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width = 2 * n_std * np.sqrt(vals[0])
    height = 2 * n_std * np.sqrt(vals[1])

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=theta,
        fill=False,
        **kwargs
    )
    ax.add_patch(ellipse)
