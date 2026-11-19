"""Reviewer-driven reanalysis from the committed score tables (no hdf5, no training).

    python scripts/reanalysis.py            # run from the repo root

Labels are reconstructed from demo index. robomimic MH files hold six
operators x 50 demos in index order; the block->tier map below is the only
one of the 90 possible assignments that reproduces the published full-set
and band AUROCs and band counts (and, for can, the tier mean durations
band.py printed from the hdf5 masks). It still deserves a direct check
against the hdf5 operator masks when the files are next on disk. can/mg holds 13 SAC checkpoints x 300
rollouts in index order (demo_1..demo_3900).
"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
N_BOOT = 2000
METRICS = ["sal", "sal_norm", "jerk", "ted", "path_len", "T"]
# 50-demo block -> tier (2 better, 1 okay, 0 worse), per task
TIER = {"can": [2, 0, 1, 2, 0, 1], "square": [2, 0, 1, 2, 0, 1]}
BANDS = {"can": [(0, 99999), (120, 220), (130, 190), (140, 175)],
         "square": [(0, 99999), (150, 300), (180, 260), (200, 240)]}


def auc(y, s):
    return roc_auc_score(y, s)


def boot_ci(y, s, n=N_BOOT):
    out = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            out.append(auc(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def delong(y, s1, s2):
    """Paired DeLong test for two correlated AUROCs. Returns (auc1, auc2, z, p)."""
    y = np.asarray(y).astype(bool)

    def comps(s):
        pos, neg = s[y], s[~y]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)

    a1, v10_1, v01_1 = comps(np.asarray(s1, float))
    a2, v10_2, v01_2 = comps(np.asarray(s2, float))
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10 / y.sum() + s01 / (~y).sum()
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, z, 2 * stats.norm.sf(abs(z))


def load_mh(path, task):
    df = pd.read_csv(path)
    df["i"] = df.demo.str.extract(r"(\d+)").astype(int)
    df = df.sort_values("i").reset_index(drop=True)
    df["op"] = df.i // 50
    return df


def section(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


# ------------------------------------------------------------------ can/mh
def mh_analysis(task, path, tiers):
    df = load_mh(path, task)
    df["skill"] = df.op.map(dict(enumerate(tiers)))
    section(f"0. {task}/mh label check (tier mean T; must match hdf5-mask output)")
    print(df.groupby("skill")["T"].mean().round(2).to_dict())
    print("per-operator T:")
    print(df.groupby(["op", "skill"])["T"].agg(["mean", "min", "max"]).round(1))

    bw = df[df.skill != 1].copy()
    bw["y"] = (bw.skill == 2).astype(int)

    section(f"1. {task}/mh operators per band (better vs worse; okay excluded)")
    for lo, hi in BANDS[task]:
        b = bw[bw["T"].between(lo, hi)]
        counts = b.groupby("op").size().to_dict()
        y = b.y.values
        line = f"T in [{lo},{hi}] n={len(b)} demos-per-operator={counts}"
        print(line)
        for m in METRICS:
            s = -b[m].values
            lo_ci, hi_ci = boot_ci(y, s)
            print(f"    {m:9s} {auc(y, s):.3f} [{lo_ci:.3f}, {hi_ci:.3f}]")

    section(f"2. {task}/mh operator-pair AUROCs (one better op vs one worse op)")
    b_ops = sorted(bw[bw.y == 1].op.unique())
    w_ops = sorted(bw[bw.y == 0].op.unique())
    for lo, hi in [BANDS[task][0], BANDS[task][-1]]:
        b = bw[bw["T"].between(lo, hi)]
        print(f"T in [{lo},{hi}]")
        for bo, wo in itertools.product(b_ops, w_ops):
            p = b[b.op.isin([bo, wo])]
            n1, n0 = int(p.y.sum()), int((1 - p.y).sum())
            if min(n1, n0) < 2:
                print(f"    better op{bo} (n={n1}) vs worse op{wo} (n={n0}): too few")
                continue
            r = "  ".join(f"{m} {auc(p.y, -p[m]):.3f}" for m in METRICS)
            print(f"    better op{bo} (n={n1}) vs worse op{wo} (n={n0}): {r}")

    section(f"3. {task}/mh leave-one-operator-out (drop one of the four operators)")
    for lo, hi in [BANDS[task][0], BANDS[task][-1]]:
        b = bw[bw["T"].between(lo, hi)]
        print(f"T in [{lo},{hi}]")
        for drop in b_ops + w_ops:
            p = b[b.op != drop]
            if p.y.nunique() < 2:
                print(f"    drop op{drop} (n={len(p)}): no worse demos left in band")
                continue
            r = "  ".join(f"{m} {auc(p.y, -p[m]):.3f}" for m in METRICS)
            print(f"    drop op{drop} (n={len(p)}): {r}")

    section(f"4. {task}/mh paired DeLong, jerk vs SAL (same demos)")
    for lo, hi in BANDS[task]:
        b = bw[bw["T"].between(lo, hi)]
        a1, a2, z, p = delong(b.y.values, -b.jerk.values, -b.sal.values)
        print(f"T in [{lo},{hi}] n={len(b)}: jerk {a1:.3f} vs SAL {a2:.3f}  "
              f"diff {a1 - a2:+.3f}  z={z:+.2f}  p={p:.4f}")
        a1, a2, z, p = delong(b.y.values, -b.jerk.values, -b["T"].values)
        print(f"{'':24s} jerk {a1:.3f} vs T   {a2:.3f}  "
              f"diff {a1 - a2:+.3f}  z={z:+.2f}  p={p:.4f}")

    section(f"5. {task}/mh length as covariate, all 200 better/worse demos")
    z = lambda v: (v - v.mean()) / v.std()
    logT = z(np.log(bw["T"])).values
    yv = bw.y.values

    def fit(X):
        """Unpenalized logistic fit; returns (log-likelihood, coefs, probs)."""
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10).fit(X, yv)
        pr = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return (yv * np.log(pr) + (1 - yv) * np.log(1 - pr)).sum(), m.coef_[0], pr

    ll0, _, p0 = fit(logT[:, None])
    print(f"length only (log T): AUROC {auc(yv, p0):.3f}  loglik {ll0:.2f}")
    for m in ["sal", "jerk", "ted", "path_len"]:
        ll1, coef, p1 = fit(np.column_stack([logT, z(bw[m]).values]))
        lr = 2 * (ll1 - ll0)
        print(f"  + {m:8s}: std. coef {coef[1]:+.2f} (negative = lower badness for "
              f"better)  LR chi2={lr:.2f} p={stats.chi2.sf(lr, 1):.4f}  "
              f"AUROC {auc(yv, p1):.3f}")
    print("partial Spearman with tier, controlling for T (rank residuals):")
    rT = stats.rankdata(bw["T"])
    for m in ["sal", "jerk", "ted", "path_len"]:
        rm, ry = stats.rankdata(bw[m]), bw.y.values.astype(float)
        res = lambda a: a - np.polyval(np.polyfit(rT, a, 1), rT)
        r, p = stats.pearsonr(res(rm), res(ry))
        print(f"  {m:8s} rho={r:+.3f} p={p:.4f}  (negative = lower badness for better)")

    section(f"6. {task}/mh length-matched pairs (greedy 1:1, |dT| <= 5 steps)")
    for seed in range(1):
        better = bw[bw.y == 1].sample(frac=1, random_state=seed)
        pool = bw[bw.y == 0].copy()
        pairs = []
        for _, r in better.iterrows():
            if pool.empty:
                break
            d = (pool["T"] - r["T"]).abs()
            j = d.idxmin()
            if d[j] <= 5:
                pairs.append((r, pool.loc[j]))
                pool = pool.drop(j)
        n = len(pairs)
        dT = np.mean([a["T"] - b["T"] for a, b in pairs])
        ops = pd.Series([f"{int(a.op)}v{int(b.op)}" for a, b in pairs]).value_counts().to_dict()
        print(f"{n} pairs, mean dT {dT:+.2f} steps, operator pairings {ops}")
        for m in ["sal", "jerk", "ted", "path_len"]:
            wins = sum(a[m] < b[m] for a, b in pairs)
            p = stats.binomtest(wins, n, 0.5).pvalue
            w = stats.wilcoxon([a[m] - b[m] for a, b in pairs]).pvalue
            print(f"  {m:8s} better demo scores lower in {wins}/{n} pairs "
                  f"({wins / n:.2f}); sign p={p:.4f}, Wilcoxon p={w:.4f}")
    return df


# ------------------------------------------------------------------ can/mg
def mg_analysis(path):
    df = pd.read_csv(path)
    df["i"] = df.demo.str.extract(r"(\d+)").astype(int)
    by_i = df.sort_values("i").reset_index(drop=True)
    by_i["ckpt"] = (by_i.i - 1) // 300
    df = df.merge(by_i[["demo", "ckpt"]], on="demo")  # keep csv order for the rng replay
    y = df.success.values

    section("7. can/mg AUROC vs success with 95% CIs (iid and checkpoint-cluster bootstrap)")
    for m in ["jerk", "sal", "path_len", "ted"]:
        s = -df[m].values
        lo, hi = boot_ci(y, s)
        cl = []
        groups = [g.index.values for _, g in df.groupby("ckpt")]
        for _ in range(N_BOOT):
            idx = np.concatenate([groups[k] for k in RNG.integers(0, 13, 13)])
            if len(np.unique(y[idx])) == 2:
                cl.append(auc(y[idx], s[idx]))
        clo, chi = np.percentile(cl, [2.5, 97.5])
        print(f"  {m:8s} {auc(y, s):.3f}  iid [{lo:.3f}, {hi:.3f}]   "
              f"cluster [{clo:.3f}, {chi:.3f}]")

    section("8. can/mg by SAC checkpoint (300 rollouts each, index order)")
    print(df.groupby("ckpt")[["success", "jerk", "sal", "ted", "path_len"]]
          .mean().round(3))
    print("\nwithin-checkpoint AUROC vs success (checkpoints with >=20 of each class):")
    rows = []
    for c, g in df.groupby("ckpt"):
        if min(g.success.sum(), (1 - g.success).sum()) >= 20:
            rows.append({"ckpt": c, "n_succ": int(g.success.sum()),
                         **{m: auc(g.success, -g[m]) for m in
                            ["jerk", "sal", "path_len", "ted"]}})
    w = pd.DataFrame(rows).set_index("ckpt")
    print(w.round(3))
    wts = w.n_succ * (300 - w.n_succ)
    print("pair-weighted mean:", {m: round(float((w[m] * wts).sum() / wts.sum()), 3)
                                  for m in ["jerk", "sal", "path_len", "ted"]})

    section("9. can/mg composition of each 1950-demo training pool")
    n_keep = len(df) // 2
    sel = {m: set(df.nsmallest(n_keep, m).demo) for m in
           ["ted", "ted_norm", "sal", "sal_norm", "jerk", "path_len", "path_len_norm"]}
    rng = np.random.default_rng(0)           # replays score.py exactly
    sel["random"] = set(rng.choice(df.demo, n_keep, replace=False))
    sel["nocuration"] = set(rng.choice(df.demo, n_keep, replace=False))
    sel["oracle"] = set(df[df.success == 1].demo)
    best = pd.read_csv("results/train_results_all.csv")
    bc = best[best.arch == "GMM-BC"].groupby("condition").best.mean()
    dp = best[best.arch != "GMM-BC"].groupby("condition").best.mean()
    print(f"{'pool':11s} {'n':>5s} {'succ':>5s} {'share':>6s} {'zero-succ ckpts 0-5':>20s} "
          f"{'ckpts 9-12':>10s} {'BC':>6s} {'DP':>6s}")
    for k in ["oracle", "nocuration", "random", "sal", "jerk", "path_len", "ted"]:
        p = df[df.demo.isin(sel[k])]
        print(f"{k:11s} {len(p):5d} {int(p.success.sum()):5d} {p.success.mean():6.3f} "
              f"{(p.ckpt <= 5).mean():20.3f} {(p.ckpt >= 9).mean():10.3f} "
              f"{bc.get(k, np.nan):6.3f} {dp.get(k, np.nan):6.3f}")
    print(f"overlap random/nocuration: {len(sel['random'] & sel['nocuration'])} demos")
    print("\nper-checkpoint demos kept (of 300):")
    tab = pd.DataFrame({k: df[df.demo.isin(sel[k])].groupby("ckpt").size()
                        for k in ["nocuration", "random", "sal", "jerk", "path_len", "ted"]}
                       ).fillna(0).astype(int)
    tab["succ_rate"] = df.groupby("ckpt").success.mean().round(2)
    print(tab)


def mg_sparc_and_jerk(path, sparc_path):
    """Numbers quoted in the paper for standard SPARC on can/mg and for the
    jerk means by outcome (run last so earlier sections keep their RNG draws)."""
    df = pd.read_csv(path).merge(pd.read_csv(sparc_path), on="demo")
    df["ckpt"] = (df.demo.str.extract(r"(\d+)")[0].astype(int) - 1) // 300
    y, s = df.success.values, -df.sparc.values
    section("10. can/mg standard SPARC: CIs, within-checkpoint, pool composition")
    lo, hi = boot_ci(y, s)
    groups = [g.index.values for _, g in df.groupby("ckpt")]
    cl = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[k] for k in RNG.integers(0, 13, 13)])
        if len(np.unique(y[idx])) == 2:
            cl.append(auc(y[idx], s[idx]))
    clo, chi = np.percentile(cl, [2.5, 97.5])
    print(f"  sparc    {auc(y, s):.3f}  iid [{lo:.3f}, {hi:.3f}]   "
          f"cluster [{clo:.3f}, {chi:.3f}]")
    w, wts = [], []
    for c, g in df.groupby("ckpt"):
        n1 = g.success.sum()
        if min(n1, len(g) - n1) >= 20:
            w.append(auc(g.success, -g.sparc))
            wts.append(n1 * (len(g) - n1))
    print(f"  within-checkpoint pair-weighted AUROC: {np.average(w, weights=wts):.3f}")
    p = df.nsmallest(len(df) // 2, "sparc")
    print(f"  SPARC top-50% pool: {len(p)} demos, {int(p.success.sum())} successes "
          f"({p.success.mean():.1%}), {(p.ckpt <= 5).mean():.1%} from checkpoints 1-6")

    section("11. can/mg mean jerk by outcome")
    for name, d in [("all checkpoints", df), ("last four checkpoints", df[df.ckpt >= 9])]:
        print(f"  {name:22s} success {d[d.success == 1].jerk.mean():.1f}   "
              f"failure {d[d.success == 0].jerk.mean():.1f}")


if __name__ == "__main__":
    mh_analysis("can", "results/scores_mh.csv", TIER["can"])
    mg_analysis("results/scores_mg.csv")
    mh_analysis("square", "results/scores_square_mh.csv", TIER["square"])
    mg_sparc_and_jerk("results/scores_mg.csv", "results/scores_mg_sparc.csv")
