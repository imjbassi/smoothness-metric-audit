"""Metric primer figure: what SAL, TED, jerk and path length reward.

ILLUSTRATIVE SYNTHETIC DATA. Nothing here is measured on robomimic.

Three synthetic pick-and-place demonstrations are built from minimum-jerk
segments between the same waypoints:
  (a) a short, smooth demonstration;
  (b) the identical motion executed twice as slowly, so the episode is
      twice as long but every segment is just as smooth;
  (c) the short demonstration with a committed correction inserted right
      before the grasp (lift, shift sideways, descend again).
A small band-limited jitter is added to every trajectory with the same
generator, standing in for operator and sensor noise; a noise-free
trajectory has an unrealistically empty spectrum and would hide the
length dependence of SAL.

Each demonstration is then scored with the project's actual metric
implementations (scripts/metrics.py) and the metrics' rankings of the
three are shown as a grid. Run from anywhere:

    python scripts/fig_metric_primer.py            # write the figure
    python scripts/fig_metric_primer.py --check    # ranking stability

Writes figures/fig_metric_primer.pdf and .png. The figure uses SEED,
chosen with --check as the seed whose relative scores are closest to the
median over seeds 0-29; --check also prints how often each ordering
occurs, which the caption relies on.
"""
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import DT, mean_jerk, path_length, sal, ted  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "figures")
SEED = 8
METRICS = ["SAL", "TED", "jerk", "path length"]
KEYS = ["a", "b", "c"]

# Colors match figure.py: SAL red, jerk blue, TED green, path length purple.
METRIC_COLORS = {"SAL": "#d62728", "TED": "#2ca02c",
                 "jerk": "#1f77b4", "path length": "#9467bd"}

GRIP_OPEN = np.array([0.04, -0.04])   # (q0, q1): width 0.08
GRIP_CLOSED = np.array([0.0, 0.0])    # width 0


def minjerk(p0, p1, n):
    """Minimum-jerk interpolation from p0 to p1 over n samples."""
    tau = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return p0 + (p1 - p0) * s


def build(segments, scale=1.0):
    """segments: list of (target_xyz or None, duration_s, gripper_state).

    A None target means dwell at the current position. The gripper state
    ('open' or 'closed') is the state reached by the end of the segment;
    a change is ramped over the first 0.3 s of that segment.
    """
    pos, grip = [], []
    cur = np.array([0.0, 0.0, 0.30])
    gstate = GRIP_OPEN.copy()
    for target, dur, gs in segments:
        n = int(round(dur * scale / DT))
        tgt = cur if target is None else np.asarray(target, float)
        pos.append(minjerk(cur, tgt, n))
        want = GRIP_OPEN if gs == "open" else GRIP_CLOSED
        g = np.tile(gstate, (n, 1)).astype(float)
        if not np.allclose(want, gstate):
            m = min(n, int(round(0.3 / DT)))
            ramp = np.linspace(0.0, 1.0, m)[:, None]
            g[:m] = gstate + (want - gstate) * ramp
            g[m:] = want
            gstate = want.copy()
        grip.append(g)
        cur = tgt
    return np.vstack(pos), np.vstack(grip)


ABOVE = (0.30, 0.00, 0.30)
GRASP = (0.30, 0.00, 0.05)
PLACE_ABOVE = (0.30, 0.35, 0.30)
PLACE = (0.30, 0.35, 0.08)

# (target, seconds, gripper state at end of segment)
PICK_PLACE = [
    (ABOVE, 2.0, "open"),          # approach
    (GRASP, 1.2, "open"),          # descend
    (None, 0.5, "closed"),         # dwell while the gripper closes
    (ABOVE, 1.2, "closed"),        # lift
    (PLACE_ABOVE, 2.0, "closed"),  # carry
    (PLACE, 1.0, "closed"),        # lower
    (None, 0.3, "open"),           # release
]

CORRECTION = [
    (ABOVE, 2.0, "open"),
    (GRASP, 1.2, "open"),
    ((0.30, 0.00, 0.15), 0.5, "open"),   # lift off: misaligned
    ((0.34, 0.04, 0.15), 0.4, "open"),   # shift sideways
    ((0.34, 0.04, 0.05), 0.5, "open"),   # descend again
    (None, 0.5, "closed"),
    ((0.34, 0.04, 0.30), 1.2, "closed"),
    (PLACE_ABOVE, 2.0, "closed"),
    (PLACE, 1.0, "closed"),
    (None, 0.3, "open"),
]


def jitter(n, rng, sigma=0.0005, width=3):
    """Band-limited positional jitter: white noise, box-smoothed."""
    w = rng.normal(0.0, sigma, size=(n + width - 1, 3))
    k = np.ones(width) / width
    return np.column_stack([np.convolve(w[:, i], k, mode="valid")
                            for i in range(3)])


def demos(seed=SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for key, segs, scale in [("a", PICK_PLACE, 1.0),
                             ("b", PICK_PLACE, 2.0),
                             ("c", CORRECTION, 1.0)]:
        pos, grip = build(segs, scale)
        pos = pos + jitter(len(pos), rng)
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(pos), 1))
        out[key] = (pos, quat, grip)
    return out


def score(pos, quat, grip):
    return {"SAL": sal(pos), "TED": ted(pos, quat, grip),
            "jerk": mean_jerk(pos), "path length": path_length(pos)}


def rank_scores(scores):
    """ranks[m][k] (1 = rated best) and rel[m][k] (score / best score)."""
    ranks, rel = {}, {}
    for m in METRICS:
        order = sorted(KEYS, key=lambda k: scores[k][m])
        ranks[m] = {k: order.index(k) + 1 for k in KEYS}
        best = scores[order[0]][m]
        rel[m] = {k: scores[k][m] / best for k in KEYS}
    return ranks, rel


def check(seeds=range(30)):
    """How stable is each metric's ordering across noise seeds?"""
    pattern = {m: Counter() for m in METRICS}
    rels = {m: {k: [] for k in KEYS} for m in METRICS}
    dist = []
    for s in seeds:
        scores = {k: score(*v) for k, v in demos(s).items()}
        _, rel = rank_scores(scores)
        for m in METRICS:
            pattern[m]["<".join(sorted(KEYS, key=lambda k: scores[k][m]))] += 1
            for k in KEYS:
                rels[m][k].append(rel[m][k])
    med = {m: {k: np.median(rels[m][k]) for k in KEYS} for m in METRICS}
    for i, s in enumerate(seeds):
        dist.append(sum(abs(np.log(rels[m][k][i]) - np.log(med[m][k]))
                        for m in METRICS for k in KEYS))
    for m in METRICS:
        print(f"{m:12s} orderings (best<...<worst): {dict(pattern[m])}")
        print(" " * 13 + "median relative score: " + "  ".join(
            f"({k}) {med[m][k]:.2f}x" for k in KEYS))
    best = list(seeds)[int(np.argmin(dist))]
    print(f"seed closest to the median: {best} (figure uses SEED = {SEED})")


def main():
    d = demos()
    scores = {k: score(*v) for k, v in d.items()}
    ranks, rel = rank_scores(scores)
    metrics, keys = METRICS, KEYS
    for m in metrics:
        print(f"{m:12s} " + "  ".join(
            f"({k}) {scores[k][m]:9.4f} rank {ranks[m][k]} "
            f"({rel[m][k]:.2f}x)" for k in keys))
    for k in keys:
        print(f"({k}) T = {len(d[k][0])} steps")

    titles = {
        "a": "(a) short and smooth",
        "b": "(b) the same motion, twice as slow",
        "c": "(c) short, with a committed correction before the grasp",
    }
    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.55, 1.0], wspace=0.28,
                          hspace=0.55, left=0.08, right=0.99,
                          top=0.90, bottom=0.16)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    tmax = max(len(v[0]) for v in d.values()) * DT
    for ax, k in zip(axes, keys):
        pos, _, grip = d[k]
        t = np.arange(len(pos) - 1) * DT
        v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / DT
        closed = np.abs(grip[:, 0] - grip[:, 1]) < 0.04
        if closed.any():
            i0, i1 = np.where(closed)[0][[0, -1]]
            ax.axvspan(i0 * DT, i1 * DT, color="0.88", lw=0, zorder=0)
        ax.plot(t, v, color="0.15", lw=1.0, zorder=3)
        if k == "c":
            # Bracket over the three correction bumps (lift, shift, descend).
            c0, c1 = 3.25, 4.65
            ax.plot([c0, c0, c1, c1], [0.40, 0.42, 0.42, 0.40], color="0.3",
                    lw=0.7, zorder=4)
            ax.text(0.5 * (c0 + c1), 0.44, "correction", fontsize=7,
                    ha="center", va="bottom", color="0.2")
        if k == "b":
            ax.text(0.5 * (i0 + i1) * DT, 0.52, "gripper closed",
                    fontsize=6.5, ha="center", va="top", color="0.35")
        ax.set_xlim(0, tmax)
        ax.set_ylim(0, 0.57)
        ax.set_yticks([0, 0.2, 0.4])
        ax.set_title(titles[k] + f"   (T = {len(pos)} steps)",
                     fontsize=8, loc="left", pad=3)
        ax.tick_params(labelsize=7)
        if k != "c":
            ax.set_xticklabels([])
    axes[1].set_ylabel("end-effector speed (m/s)", fontsize=8)
    axes[2].set_xlabel("time (s)", fontsize=8)

    # Rank grid. Cells are shaded by the score relative to the best of the
    # three under that metric, so near-ties look alike.
    axr = fig.add_subplot(gs[:, 1])
    word = {1: "1st", 2: "2nd", 3: "3rd"}
    rmax = max(max(rel[m].values()) for m in metrics)
    for i, m in enumerate(metrics):
        for j, k in enumerate(keys):
            r, x = ranks[m][k], rel[m][k]
            g = 0.95 - 0.50 * (x - 1.0) / (rmax - 1.0)
            axr.add_patch(plt.Rectangle((j, len(metrics) - 1 - i), 1, 1,
                                        facecolor=str(g), edgecolor="white",
                                        lw=1.5))
            tc = "white" if g < 0.6 else "black"
            axr.text(j + 0.5, len(metrics) - 0.38 - i, word[r], ha="center",
                     va="center", fontsize=8, color=tc)
            axr.text(j + 0.5, len(metrics) - 0.72 - i, f"{x:.2f}\u00d7",
                     ha="center", va="center", fontsize=6.5, color=tc)
        axr.text(-0.08, len(metrics) - 0.5 - i, m, ha="right", va="center",
                 fontsize=8, color=METRIC_COLORS[m], fontweight="bold")
    for j, k in enumerate(keys):
        axr.text(j + 0.5, len(metrics) + 0.12, f"({k})", ha="center",
                 va="bottom", fontsize=8)
    axr.set_xlim(0, 3)
    axr.set_ylim(0, len(metrics))
    axr.set_aspect("equal")
    axr.axis("off")
    axr.set_title("rank given by each metric\n(score relative to its best)",
                  fontsize=8, pad=14)

    fig.savefig(os.path.join(OUT, "fig_metric_primer.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_metric_primer.png"), dpi=220,
                bbox_inches="tight")
    print("wrote figures/fig_metric_primer.pdf / .png")


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        check()
    else:
        main()
