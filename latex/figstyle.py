"""Shared figure style + automatic legend placement.

Two problems this fixes, both reported from the compiled PDFs:
  1. legends sitting on top of the data or on top of an annotation;
  2. annotations pushed outside the axes frame.

`place_legend` scores the four corners by how much ink (data points, bars and
existing text) already sits there and puts the legend in the emptiest one.
`annotate_inside` clamps a text label so it can never leave the axes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, validated for colour-vision deficiency.
C = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#dcdcdc"

RC = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "legend.frameon": False, "legend.handlelength": 1.5,
    "legend.borderaxespad": 0.25, "legend.labelspacing": 0.35,
}


def use_style():
    plt.rcParams.update(RC)


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    return ax


# --------------------------------------------------------------------------
CORNERS = {
    "upper right": (0.50, 1.02, 0.55, 1.02),
    "upper left":  (-0.02, 0.50, 0.55, 1.02),
    "lower right": (0.50, 1.02, -0.02, 0.45),
    "lower left":  (-0.02, 0.50, -0.02, 0.45),
    "center right": (0.55, 1.02, 0.30, 0.70),
    "center left": (-0.02, 0.45, 0.30, 0.70),
}
ORDER = ["upper right", "upper left", "lower right", "lower left",
         "center right", "center left"]


def _axes_points(ax):
    """Every bit of drawn ink, in axes-fraction coordinates."""
    inv = ax.transAxes.inverted()
    pts = []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if len(xy):
            pts.append(inv.transform(ax.transData.transform(xy)))
    for p in ax.patches:
        try:
            bb = p.get_window_extent()
        except Exception:
            continue
        corners = np.array([[bb.x0, bb.y0], [bb.x1, bb.y0],
                            [bb.x0, bb.y1], [bb.x1, bb.y1]])
        pts.append(inv.transform(corners))
    for t in ax.texts:
        try:
            bb = t.get_window_extent()
        except Exception:
            continue
        corners = np.array([[bb.x0, bb.y0], [bb.x1, bb.y0],
                            [bb.x0, bb.y1], [bb.x1, bb.y1]])
        # text counts heavily: a legend must never land on a label
        pts.append(np.repeat(inv.transform(corners), 8, axis=0))
    for coll in ax.collections:
        try:
            off = coll.get_offsets()
        except Exception:
            continue
        if off is not None and len(off):
            pts.append(inv.transform(ax.transData.transform(np.asarray(off))))
    if not pts:
        return np.empty((0, 2))
    return np.vstack(pts)


def place_legend(ax, *args, allowed=None, headroom=None, **kw):
    """Draw the legend in the emptiest corner of `ax`.

    allowed  restrict the candidate corners (list of matplotlib loc strings)
    headroom multiply the upper y-limit by this before choosing, to make room
    """
    fig = ax.figure
    if headroom:
        lo, hi = ax.get_ylim()
        if ax.get_yscale() == "log":
            ax.set_ylim(lo, hi * headroom)
        else:
            ax.set_ylim(lo, lo + (hi - lo) * headroom)
    fig.canvas.draw()
    pts = _axes_points(ax)
    cands = [c for c in ORDER if allowed is None or c in allowed]
    best, best_score = cands[0], None
    for name in cands:
        x0, x1, y0, y1 = CORNERS[name]
        if len(pts):
            inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                      (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
            score = int(inside.sum())
        else:
            score = 0
        if best_score is None or score < best_score:
            best, best_score = name, score
    kw.setdefault("labelcolor", INK)
    return ax.legend(*args, loc=best, **kw)


def annotate_inside(ax, x, y, text, *, pad=0.015, **kw):
    """Place text at data coordinates (x, y) but never outside the axes."""
    ax.figure.canvas.draw()
    fx, fy = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
    ha = kw.pop("ha", "left")
    va = kw.pop("va", "bottom")
    if fx > 1 - pad:
        fx, ha = 1 - pad, "right"
    if fx < pad:
        fx, ha = pad, "left"
    if fy > 1 - pad:
        fy, va = 1 - pad, "top"
    if fy < pad:
        fy, va = pad, "bottom"
    return ax.text(fx, fy, text, transform=ax.transAxes, ha=ha, va=va, **kw)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
