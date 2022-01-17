import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, markers
from IPython.display import HTML
from numpy.core.fromnumeric import repeat
from celluloid import Camera
import scores

starts = [0.2] * 10
ends = np.linspace(0.4, 1.0, num=10)
masks_parameters = zip(starts, ends.tolist())
coord_range = (((-1.1, 1.1), (-1.1, 1.1)),)
latest_epoch_scorer = scores.GridScorer(20, coord_range, masks_parameters)

index = np.arange(0, 100) * 10
# fig, ax = plt.subplots(1, 2, 1, figsize=(1, 2))
fig, (ax1, ax2) = plt.subplots(1, 2)
camera = Camera(fig)

for j in [101, 152,144]:
    for i in index:
        plot = np.load(f"./checkpoints/plot{i}.npy.npz")
        ratemap = plot["arr_0"]
        auto_correlogram = plot["arr_1"]
        score_60, score_90, max_60_mask, max_90_mask, sac = zip(*[latest_epoch_scorer.get_scores(ratemap[j])])
        # Plot the activation maps
        title = "Index: %d, Epoch: %d, Gridness: %.2f" % (j, i, score_60[0])
        latest_epoch_scorer.plot_ratemap(ratemap[j], ax=ax1, cmap="jet")
        latest_epoch_scorer.plot_sac(auto_correlogram[j], mask_params=max_60_mask[0], ax=ax2, cmap="jet")
        ax1.text(0.5, -0.2, title, transform=ax1.transAxes)
        camera.snap()
anim = camera.animate(interval=100, repeat=False)
anim.save(f"ratemap_101.gif", writer="pillow")
