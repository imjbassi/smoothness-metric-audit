"""Reviewer-driven checks that need the raw robomimic files.

    python scripts/reanalysis_hdf5.py        # from the repo root

Expects datasets/{can,square}/mh/low_dim_v15.hdf5 and
datasets/{can,lift}/mg/low_dim_sparse_v15.hdf5 (v1.5 files from Hugging Face;
the committed score tables were computed from v141).
"""
import os, sys
import h5py, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score as auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import sal, mean_jerk, path_length, ted, DT

RNG = np.random.default_rng(0)
BANDS = {"can": [(0, 99999), (120, 220), (130, 190), (140, 175)],
         "square": [(0, 99999), (150, 300), (180, 260), (200, 240)]}


def section(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def sparc(pos, fs=1 / DT, padlevel=4, fc=10.0, amp_th=0.05):
    """SPARC (Balasubramanian et al. 2015) of the speed profile; returns
    BADNESS (= -SPARC). Spectrum normalized by its peak, zero-padded,
    capped at fc and at the last bin above amp_th, frequency axis rescaled
    to [0, 1], so the arc length does not grow with the number of bins."""
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fs
    nfft = int(2 ** (np.ceil(np.log2(len(v))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    M = np.abs(np.fft.fft(v, nfft))
    M = M / M.max()
    sel = f <= fc
    f, M = f[sel], M[sel]
    above = np.nonzero(M >= amp_th)[0]
    f, M = f[above[0]:above[-1] + 1], M[above[0]:above[-1] + 1]
    if len(f) < 2:
        return 0.0
    return float(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(M) ** 2).sum())


def boot(y, s, n=2000):
    out = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            out.append(auc(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def mh(task):
    path = f"datasets/{task}/mh/low_dim_v15.hdf5"
    section(f"A. {task}/mh operator masks vs index-block labels")
    with h5py.File(path, "r") as f:
        masks = {k: [d.decode() for d in f[f"mask/{k}"][:]] for k in f["mask"]}
        print("mask keys:", {k: len(v) for k, v in masks.items()})
        rows = []
        for k in f["data"]:
            p = f[f"data/{k}/obs/robot0_eef_pos"][:]
            rows.append({"demo": k, "i": int(k.split("_")[1]), "T": len(p),
                         "sal": sal(p), "sparc": sparc(p),
                         "sparc20": sparc(p, fc=20.0),
                         "jerk": mean_jerk(p), "path_len": path_length(p)})
    df = pd.DataFrame(rows).set_index("demo")
    tier = {}
    for name, v in [("worse", 0), ("okay", 1), ("better", 2)]:
        for d in masks[name]:
            tier[d] = v
    df["skill"] = pd.Series(tier)
    op_keys = sorted(k for k in masks if "operator" in k)
    for k in op_keys:
        for d in masks[k]:
            df.loc[d, "op_mask"] = k
    df["block"] = df.i // 50
    if op_keys:
        print(pd.crosstab(df.block, df.op_mask))
    print("tier by index block:")
    print(pd.crosstab(df.block, df.skill))

    old = pd.read_csv({"can": "results/scores_mh.csv",
                       "square": "results/scores_square_mh.csv"}[task]).set_index("demo")
    j = df.join(old, rsuffix="_v141")
    print(f"v15 vs v141: T identical for {(j['T'] == j['T_v141']).mean():.1%} of demos; "
          f"max |SAL diff| {np.abs(j.sal - j.sal_v141).max():.2e}; "
          f"max |jerk diff| {np.abs(j.jerk - j.jerk_v141).max():.2e}")

    section(f"B. {task}/mh SPARC with the standard frequency cap vs our SAL")
    bw = df[df.skill != 1]
    y = (bw.skill == 2).values
    print("corr with T (all 300):",
          {m: round(float(np.corrcoef(df[m], df['T'])[0, 1]), 3)
           for m in ["sal", "sparc", "sparc20", "jerk"]},
          " Spearman:",
          {m: round(float(stats.spearmanr(df[m], df['T'])[0]), 3)
           for m in ["sal", "sparc", "sparc20", "jerk"]})
    for lo, hi in BANDS[task]:
        b = bw[bw["T"].between(lo, hi)]
        yy = (b.skill == 2).values
        parts = []
        for m in ["sal", "sparc", "sparc20", "jerk", "T"]:
            l, h = boot(yy, -b[m].values)
            parts.append(f"{m} {auc(yy, -b[m]):.3f} [{l:.3f},{h:.3f}]")
        print(f"T in [{lo},{hi}] n={len(b)}: " + "  ".join(parts))
    print("operator-pair AUROC, full set (better block vs worse block):")
    for bo in sorted(bw[bw.skill == 2].block.unique()):
        for wo in sorted(bw[bw.skill == 0].block.unique()):
            p = bw[bw.block.isin([bo, wo])]
            yy = p.skill == 2
            print(f"  block{bo} vs block{wo}: " + "  ".join(
                f"{m} {auc(yy, -p[m]):.3f}" for m in ["sal", "sparc", "jerk", "T"]))


def lift_mg():
    path = "datasets/lift/mg/low_dim_sparse_v15.hdf5"
    section("C. lift/mg: does it contain failures?")
    rows = []
    with h5py.File(path, "r") as f:
        print("mask keys:", list(f["mask"].keys()) if "mask" in f else None)
        for k in f["data"]:
            g = f[f"data/{k}"]
            r = g["rewards"][:]
            o = g["obs"]
            p, q, gr = (o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
                        o["robot0_gripper_qpos"][:])
            rows.append({"demo": k, "i": int(k.split("_")[1]), "T": len(r),
                         "success": float(r[-1] > 0), "any_reward": float(r.max() > 0),
                         "sal": sal(p), "sparc": sparc(p), "jerk": mean_jerk(p),
                         "path_len": path_length(p), "ted": ted(p, q, gr)})
    df = pd.DataFrame(rows).sort_values("i").reset_index(drop=True)
    df.drop(columns="i").to_csv("results/scores_lift_mg.csv", index=False)
    print(f"{len(df)} rollouts; T min/max {df['T'].min()}/{df['T'].max()}; "
          f"final-step success {int(df.success.sum())} ({df.success.mean():.1%}); "
          f"any reward {int(df.any_reward.sum())} ({df.any_reward.mean():.1%})")
    df["ckpt"] = (df.i - df.i.min()) // 300
    print(df.groupby("ckpt")[["success", "any_reward", "jerk", "sal", "sparc",
                              "ted", "path_len"]].mean().round(3))
    y = df.success.values
    if 0 < y.mean() < 1:
        section("D. lift/mg AUROC vs success (iid CI; within-checkpoint)")
        for m in ["jerk", "sal", "sparc", "path_len", "ted"]:
            l, h = boot(y, -df[m].values)
            w = []
            for c, g in df.groupby("ckpt"):
                n1 = g.success.sum()
                if min(n1, len(g) - n1) >= 20:
                    w.append(f"c{c}:{auc(g.success, -g[m]):.3f}")
            print(f"  {m:8s} {auc(y, -df[m]):.3f} [{l:.3f}, {h:.3f}]   " + " ".join(w))


def can_mg():
    path = "datasets/can/mg/low_dim_sparse_v15.hdf5"
    if not os.path.exists(path):
        return
    section("E. can/mg: SPARC vs success, and index order = checkpoint order check")
    rows = []
    with h5py.File(path, "r") as f:
        print("mask keys:", list(f["mask"].keys()) if "mask" in f else None)
        for k in f["data"]:
            g = f[f"data/{k}"]
            p = g["obs/robot0_eef_pos"][:]
            rows.append({"demo": k, "i": int(k.split("_")[1]), "T": len(p),
                         "success": float(g["rewards"][-1] > 0),
                         "sparc": sparc(p), "sal": sal(p)})
    df = pd.DataFrame(rows)
    old = pd.read_csv("results/scores_mg.csv").set_index("demo")
    j = df.set_index("demo").join(old, rsuffix="_v141")
    print(f"success label identical to v141 for {(j.success == j.success_v141).mean():.1%}; "
          f"max |SAL diff| {np.abs(j.sal - j.sal_v141).max():.2e}")
    y = df.success.values
    for m in ["sal", "sparc"]:
        l, h = boot(y, -df[m].values)
        print(f"  {m:6s} AUROC {auc(y, -df[m]):.3f} [{l:.3f}, {h:.3f}]")


if __name__ == "__main__":
    mh("can")
    mh("square")
    lift_mg()
    can_mg()
