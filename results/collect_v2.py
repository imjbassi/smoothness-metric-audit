"""Collect the v2 GMM-BC grids (scripts/gen_configs_v2.py) and test them.

    python results/collect_v2.py --root <results_train_v2>      # parse logs
    python results/collect_v2.py --from-csv results/train_results_v2.csv

Same estimator as collect2.py (best of the eight rollout evaluations per
run), with the mean of the eight reported alongside. Old can/mg BC arms
come from results/train_results_all.csv.
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect2 import parse_run  # noqa: E402

RUN_RE = re.compile(r"^(?P<task>mh|mg)_(?P<cond>[a-z]+)_s(?P<seed>\d+)$")
HERE = os.path.dirname(os.path.abspath(__file__))


def load(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        m = RUN_RE.match(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "DONE")):
            continue
        hits = parse_run(d)
        if hits:
            rows.append({"task": m["task"], "condition": m["cond"],
                         "seed": int(m["seed"]), "best": max(hits),
                         "mean8": float(np.mean(hits)), "n_evals": len(hits)})
    return pd.DataFrame(rows)


def show(name, v):
    print(f"  {name:22s} best {v.best.mean():.3f} +/- {v.best.std(ddof=1):.3f}"
          f"   mean-of-evals {v.mean8.mean():.3f}   n={len(v)}")


def tests(pairs, g, label):
    alpha = 0.05 / len(pairs)
    print(f"\n  {label}: Welch t-tests, Bonferroni alpha = {alpha:.4f}")
    for a, b in pairs:
        for est in ["best", "mean8"]:
            x, y = g[a][est].values, g[b][est].values
            if len(x) < 2 or len(y) < 2 or np.isnan(y).all():
                continue
            t, p = stats.ttest_ind(x, y, equal_var=False)
            print(f"    {a:16s} vs {b:16s} [{est:5s}] diff {x.mean() - y.mean():+.3f}"
                  f"  p={p:.4f} {'SURVIVES' if p < alpha else ''}")


def main(a):
    df = pd.read_csv(a.from_csv) if a.from_csv else load(a.root)
    if not a.from_csv:
        df.to_csv(os.path.join(HERE, "train_results_v2.csv"), index=False)
    print(f"{len(df)} finished runs")

    mh = {c: df[(df.task == "mh") & (df.condition == c)] for c in
          ["sal", "length", "random"]}
    print("\ncan/mh, top 150 of 300 demonstrations (GMM-BC)")
    for c, v in mh.items():
        if len(v):
            show(c, v)
    if all(len(v) > 1 for v in mh.values()):
        tests([("sal", "random"), ("length", "random"), ("sal", "length")],
              mh, "can/mh")

    old = pd.read_csv(os.path.join(HERE, "train_results_all.csv"))
    old = old[old.arch == "GMM-BC"]
    g = {f"old_{c}": old[old.condition == c].assign(mean8=np.nan)
         for c in ["random", "nocuration", "ted", "jerk", "sal"]}
    for c in ["sparc", "all", "random"]:
        g[f"new_{c}"] = df[(df.task == "mg") & (df.condition == c)]
    print("\ncan/mg (GMM-BC); old_* = original grid, new_* = rebuilt environment")
    for k, v in g.items():
        if len(v):
            show(k, v)
    if len(g["new_random"]) > 1:
        seeds = g["new_random"].seed.values
        o = g["old_random"][g["old_random"].seed.isin(seeds)].best.values
        print(f"\n  anchor: random seeds {sorted(seeds)} new {g['new_random'].best.mean():.3f}"
              f" vs original {o.mean():.3f} (same seeds, same filter key)")
    pairs = [(a_, b_) for a_, b_ in [("new_sparc", "old_random"),
                                    ("new_sparc", "old_nocuration"),
                                    ("new_all", "old_random"),
                                    ("new_all", "old_nocuration"),
                                    ("new_sparc", "new_all")]
             if len(g[a_]) > 1 and len(g[b_]) > 1]
    if pairs:
        tests(pairs, g, "can/mg")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root")
    p.add_argument("--from-csv")
    main(p.parse_args())
