"""Collect results and run the corrected statistical comparison.

Handles one or both architectures. Reports, for every metric-curated
condition, the difference against each baseline with a bootstrap CI on the
difference and a Welch two-sample p-value, plus Bonferroni correction across
all reported comparisons. This replaces eyeballing CI overlap, which is a
conservative test and understates what is distinguishable.

Usage:
    python collect2.py                              # GMM-BC only
    python collect2.py --root results_train_dp      # diffusion policy only
    python collect2.py --root results_train --root2 results_train_dp
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy import stats

RUN_RE = re.compile(r"^(?P<cond>[a-z_]+)_s(?P<seed>\d+)$")
BASELINES = ["random", "nocuration"]
N_BOOT = 20000


def parse_run(run_dir):
    hits = []
    for p in glob.glob(os.path.join(run_dir, "*", "models", "*success*.pth")):
        m = re.search(r"model_epoch_(\d+).*?success_([0-9.]+)\.pth$",
                      os.path.basename(p))
        if m:
            hits.append(float(m.group(2).rstrip(".")))
    if hits:
        return hits
    for p in glob.glob(os.path.join(run_dir, "*", "logs", "log.txt")):
        txt = open(p, errors="ignore").read()
        hits += [float(m.group(1))
                 for m in re.finditer(r'"Success_Rate":\s*([0-9.]+)', txt)]
    return hits


def load(root, label):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
        m = RUN_RE.match(os.path.basename(run_dir))
        if not m:
            continue
        hits = parse_run(run_dir)
        if not hits:
            continue
        rows.append({"arch": label, "condition": m.group("cond"),
                     "seed": int(m.group("seed")), "best": max(hits)})
    return pd.DataFrame(rows)


def diff_ci(a, b, rng):
    bs = [rng.choice(a, len(a), replace=True).mean()
          - rng.choice(b, len(b), replace=True).mean()
          for _ in range(N_BOOT)]
    return np.percentile(bs, [2.5, 97.5])


def report(df, label):
    print(f"\n{'='*66}\n{label}  ({len(df)} runs)\n{'='*66}")
    rng = np.random.default_rng(0)
    g = {c: df[df.condition == c].best.values for c in df.condition.unique()}

    order = ["oracle"] + BASELINES + \
            [c for c in ["sal", "jerk", "path_len", "ted"] if c in g]
    print("\nmean best success rate")
    for c in [c for c in order if c in g]:
        v = g[c]
        boot = [rng.choice(v, len(v), replace=True).mean()
                for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {c:11s} {v.mean():.3f} +/- {v.std(ddof=1):.3f} "
              f"[{lo:.3f}, {hi:.3f}]   n={len(v)}")

    metrics = [c for c in ["sal", "jerk", "path_len", "ted"] if c in g]
    tests = []
    for base in BASELINES:
        if base not in g:
            continue
        for m in metrics:
            lo, hi = diff_ci(g[m], g[base], rng)
            t, p = stats.ttest_ind(g[m], g[base], equal_var=False)
            tests.append((m, base, g[m].mean() - g[base].mean(), lo, hi, p))

    if not tests:
        return
    alpha = 0.05 / len(tests)
    print(f"\npairwise tests ({len(tests)} comparisons, "
          f"Bonferroni alpha = {alpha:.4f})")
    print(f"  {'metric':11s} {'vs':12s} {'diff':>7s}  "
          f"{'95% CI of diff':>20s}  {'p':>8s}  survives")
    for m, base, d, lo, hi, p in tests:
        star = "YES" if p < alpha else ""
        print(f"  {m:11s} {base:12s} {d:+7.3f}  "
              f"[{lo:+6.3f},{hi:+6.3f}]  {p:8.4f}  {star}")

    if all(b in g for b in BASELINES):
        lo, hi = diff_ci(g["random"], g["nocuration"], rng)
        d = g["random"].mean() - g["nocuration"].mean()
        print(f"\n  sanity: random vs nocuration {d:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] "
              f"({'indistinguishable' if lo < 0 < hi else 'DIFFERENT'})")


def main(a):
    frames = []
    if os.path.isdir(a.root):
        frames.append(load(a.root, "GMM-BC"))
    if a.root2 and os.path.isdir(a.root2):
        frames.append(load(a.root2, "diffusion"))
    if not frames or all(f.empty for f in frames):
        print("No completed runs found.")
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("train_results_all.csv", index=False)

    for arch in df.arch.unique():
        report(df[df.arch == arch], arch)

    if df.arch.nunique() > 1:
        print(f"\n{'='*66}\ncross-architecture: does the ordering hold?\n{'='*66}")
        piv = df.pivot_table(index="condition", columns="arch",
                             values="best", aggfunc="mean").round(3)
        print(piv)
        print("\nIf TED is the worst condition under both architectures, the")
        print("finding is a property of the metric, not of behavior cloning.")

    print("\nwrote train_results_all.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_train")
    p.add_argument("--root2", default=None)
    main(p.parse_args())
