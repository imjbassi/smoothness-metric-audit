"""Configs for the v2 GMM-BC grids; identical training settings to the
original can/mg grid (gen_configs.base), 10 seeds each.

    python scripts/gen_configs_v2.py --data /root/data --out results_train_v2

Writes configs_v2/<task>_<cond>_s<seed>.json and configs_v2/jobs.txt
(one config path per line, anchor runs first so a broken environment
shows up early). Run with scripts/run_grid_v2.sh.
"""
import argparse, copy, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_configs import base  # noqa: E402

SEEDS = list(range(10))
GRIDS = [
    # (task, hdf5 relative to --data, filter key, seeds)
    ("mg", "can/mg/low_dim_sparse_v141.hdf5", "random_top50", SEEDS[:5]),  # anchor
    ("mh", "can/mh/low_dim_v141.hdf5", "sal_top50", SEEDS),
    ("mh", "can/mh/low_dim_v141.hdf5", "length_top50", SEEDS),
    ("mh", "can/mh/low_dim_v141.hdf5", "random_top50", SEEDS),
    ("mg", "can/mg/low_dim_sparse_v141.hdf5", "sparc_top50", SEEDS),
    ("mg", "can/mg/low_dim_sparse_v141.hdf5", "all_top100", SEEDS),
]


def main(a):
    os.makedirs("configs_v2", exist_ok=True)
    # interleave by seed so a partial grid is still balanced across arms
    jobs = []
    for s in SEEDS:
        for task, rel, key, seeds in GRIDS:
            if s not in seeds:
                continue
            cfg = copy.deepcopy(base(os.path.join(a.data, rel), a.out))
            cfg["train"]["output_dir"] = os.path.join(a.root, a.out)
            name = f"{task}_{key.split('_top')[0]}_s{s}"
            cfg["experiment"]["name"] = name
            cfg["train"]["hdf5_filter_key"] = key
            cfg["train"]["seed"] = s
            path = f"configs_v2/{name}.json"
            with open(path, "w") as f:
                json.dump(cfg, f, indent=4)
            jobs.append(path)
    with open("configs_v2/jobs.txt", "w", newline="\n") as f:
        f.write("\n".join(jobs) + "\n")
    print(f"wrote {len(jobs)} configs and configs_v2/jobs.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/root/data")
    p.add_argument("--root", default="/root/robomimic")
    p.add_argument("--out", default="results_train_v2")
    main(p.parse_args())
