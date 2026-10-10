"""Velocity-plane diagnostics for BKW snapshots (NumPy/Matplotlib only)."""

from pathlib import Path

import numpy as np


PLANE_FIELDS = ("plane_density_estimated", "plane_density_exact",
                "plane_score_estimated", "plane_score_exact")
PLANE_METADATA = ("plane_grid", "plane_pairs")


def save_plane_plots(outdir, snapshots, *, status_note=None):
    """One figure per snapshot; shared color and arrow scales across time/planes.

    Colors for scores use every velocity component. Arrows show only the two
    components in the displayed plane. Old archives without plane data skip
    these figures rather than substituting a KDE score for a saved network.
    """
    snapshots = [s for s in snapshots if all(k in s for k in (*PLANE_METADATA, *PLANE_FIELDS))]
    if not snapshots:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    density_max = max(float(np.max(s[k])) for s in snapshots for k in PLANE_FIELDS[:2])
    density_error_max = max(float(np.max(np.abs(s[PLANE_FIELDS[0]] - s[PLANE_FIELDS[1]])))
                            for s in snapshots)
    score_max = max(float(np.max(np.linalg.norm(s["plane_score_estimated"], axis=-1))) for s in snapshots)
    score_error_max = max(float(np.max(np.linalg.norm(
        s["plane_score_estimated"] - s["plane_score_exact"], axis=-1))) for s in snapshots)
    # Nonzero ranges keep exact/zero-error panels and their colorbars well-defined.
    norms = [Normalize(0, density_max or 1),
             Normalize(-(density_error_max or 1e-15), density_error_max or 1e-15),
             Normalize(0, score_max or 1), Normalize(0, score_error_max or 1e-15)]
    cmaps = ["viridis", "RdBu_r", "viridis", "magma"]
    titles = ["KDE density + BKW contours", "Density difference: KDE − BKW",
              "Computed score", "Score error: computed − BKW"]
    color_labels = ["Density", "Signed density difference", "Full score norm", "Full score error norm"]
    targets = []
    for index, snap in enumerate(snapshots):
        grid, pairs = snap["plane_grid"], snap["plane_pairs"]
        x, y = np.meshgrid(grid, grid, indexing="xy")
        estimated, exact = snap["plane_density_estimated"], snap["plane_density_exact"]
        score = snap["plane_score_estimated"]
        error = score - snap["plane_score_exact"]
        fields = [estimated, estimated - exact, np.linalg.norm(score, axis=-1), np.linalg.norm(error, axis=-1)]
        fig, axes = plt.subplots(len(pairs), 4, figsize=(17, 4.1 * len(pairs)),
                                 squeeze=False, layout="constrained")
        stride = max(1, len(grid) // 9)
        arrow_spacing = (grid[-1] - grid[0]) * stride / (len(grid) - 1)
        for row, (a, b) in enumerate(pairs):
            for col in range(4):
                ax = axes[row, col]
                mesh = ax.pcolormesh(x, y, fields[col][row], shading="nearest",
                                     cmap=cmaps[col], norm=norms[col], rasterized=True)
                if col == 0:
                    levels = np.linspace(0, density_max, 6)[1:-1]
                    levels = levels[(levels > exact[row].min()) & (levels < exact[row].max())]
                    if len(levels):
                        ax.contour(x, y, exact[row], levels=levels, colors="white", linewidths=0.7)
                if col in (2, 3):
                    vector = score[row] if col == 2 else error[row]
                    maximum = score_max if col == 2 else score_error_max
                    if maximum > 0:
                        ax.quiver(x[::stride, ::stride], y[::stride, ::stride],
                                  vector[::stride, ::stride, a], vector[::stride, ::stride, b],
                                  angles="xy", scale_units="xy", scale=maximum / (0.8 * arrow_spacing),
                                  color="white", width=0.004)
                ax.set(xlabel=rf"$v_{a+1}$", ylabel=rf"$v_{b+1}$", aspect="equal",
                       xlim=(grid[0], grid[-1]), ylim=(grid[0], grid[-1]))
                if row == 0:
                    ax.set_title(titles[col], fontsize=11)
                if row == len(pairs) - 1:
                    fig.colorbar(mesh, ax=axes[:, col].tolist(), label=color_labels[col],
                                 location="bottom", shrink=0.85, pad=0.07)
        title = (f"BKW velocity-plane slices | t={snap['time']:.6g}\n"
                 "Unshown velocity coordinates = 0; arrows = in-plane components; "
                 "color and arrow scales shared across saved times")
        if status_note:
            title += "\n" + status_note
        fig.suptitle(title, fontsize=12)
        target = Path(outdir) / f"density_score_planes_{index:03d}_t{snap['time']:.6g}.png"
        fig.savefig(target, dpi=160, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        targets.append(target)
    return targets
