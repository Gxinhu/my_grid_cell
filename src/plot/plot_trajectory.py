# %%

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, markers
from IPython.display import HTML
from numpy.core.fromnumeric import repeat
from celluloid import Camera

plot = np.load("./checkpoints/plot990.npy.npz")
trajectory = np.load("./checkpoints/trajectory990.npy.npz")

real_position = trajectory["arr_0"]
predition_position = trajectory["arr_1"]
predition_position[:, 0] = real_position[:, 0]
error = np.absolute(real_position - predition_position)
error_index = np.argpartition(np.sum(np.sum(error, 2), 1), 3)

# %%
for i in error_index[:20]:
    fig, ax = plt.subplots(figsize=(8, 8))
    pick_traj = i
    ax.plot(real_position[pick_traj][:, 0], real_position[pick_traj][:, 1], "r--")
    ax.set(xlim=[-1.1, 1.1], ylim=[-1.1, 1.1])
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("Y", fontsize=15)

    camera = Camera(fig)
    for j in range(1, real_position.shape[1] + 1):
        x = real_position[pick_traj][0:j, 0]
        y = real_position[pick_traj][0:j, 1]
        x_pred = predition_position[pick_traj][0:j, 0]
        y_pred = predition_position[pick_traj][0:j, 1]
        ax.plot(x[-1], y[-1], marker="o", markersize=12, markeredgecolor="r", markerfacecolor="r")
        (line1,) = ax.plot(x, y, color="b", lw=2, linestyle="--")
        if j == 1:
            line1.set_label("real trajectory")
        ax.plot(x_pred[-1], y_pred[-1], marker="o", markersize=12, markeredgecolor="g", markerfacecolor="g")
        (line2,) = ax.plot(x_pred, y_pred, color="y", lw=2, linestyle="--")
        if j == 1:
            line2.set_label("predict trajectory")
        ax.set(xlim=[-1.1, 1.1], ylim=[-1.1, 1.1])
        camera.snap()
    ax.legend()
    anim = camera.animate(interval=200, repeat=True, repeat_delay=4000)
    anim.save(f"trajectory_compare_{i}.gif", writer="pillow")

# %%
