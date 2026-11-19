# Smoothness Ranks Skill, Not Success

An audit of published trajectory-smoothness curation metrics (SAL and TED,
from *rinse*, arXiv:2604.23000) under episode-length controls, on real
robomimic operator labels and task-success labels.

**Headline results**

| setting | jerk | SAL | TED | episode length |
|---|---|---|---|---|
| can/mh, no length control (skill) | 0.584 | 0.952 | 0.558 | 0.957 |
| can/mh, duration-matched (skill)  | 0.896 | 0.628 | 0.359 | 0.485 |
| can/mg, T fixed at 150 (success)  | 0.706 | 0.454 | 0.376 | 0.500 |

Detection AUROC. Above 0.500 = ranks quality correctly; below = inverted.

Downstream (GMM-BC, 10 seeds): filtering can/mg by TED produces a **worse**
policy than a random subset of the same size (0.240 vs. 0.360 and 0.380 for
two independent random 50% draws). The second draw is called `nocuration` in
the configs and CSVs; it is not the full 3900-demonstration pool.

## Layout

```
scripts/     metric implementations, scoring, analysis, figures
paper/       LaTeX source
figures/     generated figures (PDF for LaTeX, PNG for preview)
results/     scored CSVs and analysis output
configs/     robomimic training configs (35 = 7 conditions x 5 seeds)
```

## Environment

Windows is not viable for this stack (robosuite has no official Windows
support, `egl_probe` has no Windows build). Use WSL2 or Linux.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install "mujoco==2.3.2"          # install BEFORE robosuite
pip install "robosuite==1.4.1"
pip install -e .                     # robomimic, cloned separately
pip install h5py scipy pandas scikit-learn fastdtw matplotlib
export MUJOCO_GL=egl
```

Python 3.12+ will not work: `mujoco` 2.x ships no wheels for it, and
`robosuite` 1.4.1 asserts against MuJoCo 3.x joint-type enums.

## Reproducing

```bash
# 1. data
python robomimic/scripts/download_datasets.py --tasks can \
    --dataset_types mh mg paired --hdf5_types low_dim

# 2. score all demonstrations, write filter keys into the hdf5
python scripts/score.py --hdf5 datasets/can/mg/low_dim_sparse_v141.hdf5 --keep 0.5
python scripts/score.py --hdf5 datasets/can/mh/low_dim_v141.hdf5 --keep 0.5

# 3. detection analysis
python scripts/skill.py results/scores_mh.csv datasets/can/mh/low_dim_v141.hdf5
python scripts/band.py  results/scores_mh.csv datasets/can/mh/low_dim_v141.hdf5 140 175

# 4. training grid (35 runs, ~27h on one GPU)
python scripts/gen_configs.py --hdf5 datasets/can/mg/low_dim_sparse_v141.hdf5
nohup bash run_all.sh > train_log.txt 2>&1 &

# 5. collect and plot
python scripts/collect.py
python scripts/figure.py
python scripts/qualfig2.py --hdf5 datasets/can/mg/low_dim_sparse_v141.hdf5 \
    --success demo_1832 --failure demo_753
```

`resume.sh` reruns only incomplete runs, for when the grid is interrupted.

## Caveats

TED is reimplemented from the *rinse* appendix, not the authors' released
code. SAL follows the published formula. Both are in `scripts/metrics.py`
and should be checked against reference implementations if those become
available.

## Data

All datasets are public robomimic (Mandlekar et al., 2021). No proprietary
or employer data is used anywhere in this project.
