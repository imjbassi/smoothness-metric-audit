"""Write the filter keys for the v2 training grids into robomimic v141 files.

    python scripts/prep_grids_v2.py --mh  <can/mh/low_dim_v141.hdf5> \
                                    --mg  <can/mg/low_dim_sparse_v141.hdf5>

can/mh (all 300 demos, keep 150; the reviewer-requested experiment):
    sal_top50      lowest RINSE SAL
    length_top50   shortest episodes
    random_top50   uniform random (default_rng(0))
can/mg (3900 rollouts):
    sparc_top50    lowest SPARC badness (standard SPARC), 1950
    all_top100     every rollout, 3900 (true "no curation")
    random_top50   re-created exactly as score.py drew it, so that the
                   original random arm can be re-run in the rebuilt
                   environment as a reproducibility anchor
Also checks that the can/mg random draw reproduces the committed pool
(344 successes) and that SPARC matches results/scores_mg_sparc.csv.
"""
import argparse, csv, os, sys
import h5py, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import sal                      # noqa: E402
from reanalysis_hdf5 import sparc            # noqa: E402


def put(f, name, demos):
    if f"mask/{name}" in f:
        del f[f"mask/{name}"]
    f.create_dataset(f"mask/{name}", data=np.array(sorted(demos), dtype="S"))
    print(f"  wrote mask/{name}: {len(demos)} demos")


def mh(path):
    print(f"can/mh: {path}")
    with h5py.File(path, "a") as f:
        keys = list(f["data"].keys())
        T = {k: f[f"data/{k}/actions"].shape[0] for k in keys}
        S = {k: sal(f[f"data/{k}/obs/robot0_eef_pos"][:]) for k in keys}
        better = {d.decode() for d in f["mask/better"][:]}
        worse = {d.decode() for d in f["mask/worse"][:]}
        n = len(keys) // 2
        sel = {"sal": sorted(keys, key=S.get)[:n],
               "length": sorted(keys, key=lambda k: (T[k], k))[:n],
               "random": list(np.random.default_rng(0).choice(keys, n, replace=False))}
        for c, d in sel.items():
            d = set(d)
            print(f"  {c:7s} better {len(d & better):3d}  worse {len(d & worse):3d}  "
                  f"okay {len(d - better - worse):3d}  mean T {np.mean([T[k] for k in d]):.1f}")
            put(f, f"{c}_top50", d)
        print(f"  SAL top half vs length top half overlap: "
              f"{len(set(sel['sal']) & set(sel['length']))}/{n}")


def mg(path):
    print(f"can/mg: {path}")
    ref = {r["demo"]: float(r["sparc"]) for r in
           csv.DictReader(open(os.path.join(HERE, "..", "results", "scores_mg_sparc.csv")))}
    with h5py.File(path, "a") as f:
        keys = list(f["data"].keys())
        succ = {k: float(f[f"data/{k}/rewards"][-1] > 0) for k in keys}
        sp = {k: sparc(f[f"data/{k}/obs/robot0_eef_pos"][:]) for k in keys}
        print(f"  SPARC max |diff| vs scores_mg_sparc.csv: "
              f"{max(abs(sp[k] - ref[k]) for k in keys):.2e}")
        n = len(keys) // 2
        rng = np.random.default_rng(0)
        rand = list(rng.choice(keys, n, replace=False))   # score.py: "random"
        sel = {"sparc_top50": sorted(keys, key=sp.get)[:n],
               "all_top100": keys,
               "random_top50": rand}
        for name, d in sel.items():
            print(f"  {name:13s} successes {int(sum(succ[k] for k in d))}")
            put(f, name, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mh")
    p.add_argument("--mg")
    a = p.parse_args()
    if a.mh:
        mh(a.mh)
    if a.mg:
        mg(a.mg)
