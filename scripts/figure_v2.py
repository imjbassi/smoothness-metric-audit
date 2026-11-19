"""Main figure for the arXiv v2 (paper.tex): figures/fig_main_v2.pdf.

Same as figure.py (which the workshop versions still use) plus standard
SPARC in every panel and a third panel for lift/mg.

Panel A: can/mh better-vs-worse AUROC across four duration bands
         (shaded = 95% bootstrap CI).
Panel B: can/mg (T = 150 for every rollout), AUROC against task success;
         error bars = 95% checkpoint-cluster bootstrap CI.
Panel C: lift/mg (T = 150), AUROC against final-step success;
         error bars = 95% iid bootstrap CI (5 checkpoints are too few to
         cluster).

Numbers come from results/reanalysis.txt (scripts/reanalysis.py) and
results/reanalysis_hdf5.txt (scripts/reanalysis_hdf5.py).

    python scripts/figure_v2.py        # from the repo root
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
C = {"SAL": "#d62728", "SPARC": "#ff7f0e", "jerk": "#1f77b4",
     "TED": "#2ca02c", "path len": "#9467bd", "length": "#7f7f7f"}

# ---- A: can/mh, four bands (reanalysis_hdf5.txt section B; TED from band.py)
bands = ["full\n(n=200)", "120–220\n(n=98)", "130–190\n(n=74)", "140–175\n(n=40)"]
x = np.arange(4)
series = {
    "SAL":   ([0.952, 0.793, 0.759, 0.628], [0.922, 0.690, 0.628, 0.378],
              [0.975, 0.881, 0.875, 0.856], "o"),
    "SPARC": ([0.764, 0.541, 0.568, 0.506], [0.696, 0.415, 0.417, 0.270],
              [0.825, 0.666, 0.713, 0.762], "v"),
    "jerk":  ([0.584, 0.862, 0.906, 0.896], [0.505, 0.783, 0.832, 0.783],
              [0.665, 0.928, 0.965, 0.980], "s"),
    "TED":   ([0.558, 0.653, 0.631, 0.359], [0.480, 0.522, 0.468, 0.147],
              [0.636, 0.776, 0.779, 0.583], "^"),
    "length": ([0.957, 0.801, 0.753, 0.485], [0.929, 0.689, 0.594, 0.227],
               [0.980, 0.901, 0.891, 0.740], "D"),
}

# ---- B: can/mg vs success, cluster CIs (reanalysis.txt section 7) ----------
mg = [("jerk", 0.706, 0.573, 0.832), ("SPARC", 0.871, 0.817, 0.907),
      ("SAL", 0.454, 0.392, 0.527), ("path len", 0.421, 0.284, 0.563),
      ("TED", 0.376, 0.300, 0.447)]

# ---- C: lift/mg vs success, iid CIs (reanalysis_hdf5.txt section D) -------
lift = [("jerk", 0.870, 0.843, 0.894), ("SPARC", 0.891, 0.861, 0.919),
        ("SAL", 0.643, 0.607, 0.679), ("path len", 0.431, 0.401, 0.462),
        ("TED", 0.669, 0.639, 0.700)]

fig, (a1, a2, a3) = plt.subplots(
    1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})

for name, (v, lo, hi, m) in series.items():
    ls = "--" if name == "length" else "-"
    lab = "episode length (T)" if name == "length" else name
    a1.plot(x, v, ls, marker=m, ms=4, lw=1.5, color=C[name], label=lab, zorder=3)
    a1.fill_between(x, lo, hi, color=C[name], alpha=0.10, lw=0)
a1.axhline(0.5, color="k", lw=0.8, ls=":", zorder=1)
a1.set_xticks(x)
a1.set_xticklabels(bands, fontsize=7)
a1.set_ylim(0.05, 1.02)
a1.set_ylabel("AUROC (better vs. worse operator)", fontsize=8)
a1.set_xlabel("episode-length band (tighter →)", fontsize=8)
a1.set_title("(a) can/mh: operator labels", fontsize=8.5)
a1.legend(fontsize=6.3, loc="lower left", framealpha=0.9, ncol=2,
          handlelength=1.8, columnspacing=0.8)
a1.tick_params(labelsize=7.5)


def bars(ax, rows, title):
    xs = np.arange(len(rows))
    v = np.array([r[1] for r in rows])
    err = np.array([[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]])
    ax.bar(xs, v, color=[C[r[0]] for r in rows], width=0.62,
           yerr=err, error_kw=dict(lw=0.8, capsize=2, ecolor="k"))
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    for xi, r in zip(xs, rows):
        ax.text(xi, r[3] + 0.02, f"{r[1]:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7.5)


bars(a2, mg, "(b) can/mg: success, T=150")
a2.set_ylabel("AUROC (task success)", fontsize=8)
bars(a3, lift, "(c) lift/mg: success, T=150")

fig.tight_layout(w_pad=0.6)
fig.savefig(os.path.join(OUT, "fig_main_v2.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_main_v2.png"), dpi=220, bbox_inches="tight")
print("wrote figures/fig_main_v2.pdf / .png")
