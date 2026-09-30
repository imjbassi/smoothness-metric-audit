# Project handoff: smoothness-metric-audit (as of 2026-09-29)

## What this is
An audit of trajectory-smoothness curation metrics for robot imitation learning. The target is RINSE (Kulkarni et al. 2026, arXiv:2604.23000), which proposes two metrics:
- **SAL**: Spectral Arc Length.
- **TED**: a contact-aware Trajectory-Envelope Distance.

It claims that filtering demonstrations by these metrics improves downstream policies. **Core finding: TED cannot tell real failures from successes, and curating by it produces worse behavior-cloning policies than not curating at all.** The smoothness metrics mostly measure episode duration. Raw jerk, the crude baseline RINSE argues against, beats them once duration is controlled.

## Detection results (robomimic low-dim, AUROC with 95% bootstrap CIs)
- **can/mh**, operator skill tiers:
  - SAL 0.952 vs episode length alone 0.957; SAL correlates +0.97 with T.
  - Tightest duration band (T∈[140,175], n=40, length at chance 0.485): SAL 0.628, jerk 0.896, TED 0.359.
  - Per-timestep normalized SAL: 0.582.
- **square/mh**, no retuning: SAL 0.898→0.480 and jerk 0.431→0.896 as the length control goes 0.936→0.407; SAL correlates +0.96 with T.
- **can/mg**, every episode exactly 150 steps (no length confound), 716 of 3900 successful: against task success, jerk 0.706, SAL 0.454, path length 0.421, **TED 0.376 (inverted)**.
- **can/paired**, secondary: every metric inverted, but with the opposite length confound (length AUROC 0.194). Its failures are *staged* by an operator.
- Mechanism: in can/mg, failure looks like smooth hovering and success needs high-deviation contact motion, so smoothness measures commitment with the sign flipped.

## Downstream training on can/mg (50% keep; pools of 1950; oracle pool 716)
**GMM-BC**: MLP 1024×1024, 5 modes, batch 100, 600 epochs × 500 steps, lr 1e-4, 10 seeds.
- Success: oracle 0.892, no curation 0.380, random 0.360, SAL 0.332, jerk 0.332, path length 0.310, **TED 0.240**.
- TED vs random −0.120 (p=0.0001); vs no curation −0.140 (p<0.0001). Both survive Bonferroni across 8 tests (α=0.00625).
- Path length: p=0.037 / 0.0071, does not survive. SAL and jerk: p≥0.06, not distinguishable.

**Diffusion policy**: robomimic conditional U-Net [256,512,1024], kernel 5, 8 groups, embed 256, EMA 0.75, obs/action/pred horizons 2/8/16, DDIM 10 steps, batch 256, same epochs, lr 1e-4. 5 conditions × 5 seeds (SAL and path length not run).
- Success: oracle 0.544, no curation 0.120, random 0.116, **jerk 0.212**, **TED 0.092**.
- Jerk vs random +0.096 (p=0.0007), vs no curation +0.092 (p=0.0011). Both survive α=0.0125 (4 tests). **Jerk reverses sign** relative to BC.
- TED vs random −0.024 (p=0.17), vs no curation −0.028 (p=0.11). Not resolvable: the grid's minimum detectable difference is about 0.054 at n=5. **TED still ranks worst.**
- Pooling all 12 tests (α=0.00417) changes no conclusion.
- Absolute gap: diffusion is 1.6× lower on the oracle and 3.1× lower on the baselines.
- Relative deficit, descriptive only: TED is 37% below no curation under BC and 23% under diffusion.
- Estimator: results are best checkpoint of 8 evaluations, in both grids. With mean-of-8 instead, TED vs no curation under diffusion gives p=0.0110, which *would* survive. The papers report the conservative version and say so in a footnote or appendix.

## Why is diffusion policy so much lower? (diagnostics)
- **D1, training curves:** the diffusion runs don't learn after epoch 100. First-half to second-half gain is +0.017 against a 0.044 noise floor; 3 of 25 runs improve; sd/noise ratio 1.01. BC gains +0.092, 56 of 70 improve, ratio 1.46. **Under-training is ruled out.**
- **D2, denoising steps:** DDIM at 10/50/100 steps gives 0.42/0.46/0.42. **Sampling budget is ruled out.**
- **Chi et al. 2023, Diffusion Policy (arXiv:2303.04137, verified from the PDF):**
  - Table 1 reports "(max over checkpoints)/(avg of last 10)" over 3 seeds × 50 initializations, with ~90 checkpoints (every 50 of 4500 epochs).
  - Can (ph): DP-C 1.00/0.96, DP-T 1.00/1.00, LSTM-GMM 1.00/0.91.
  - Their diffusion policy uses **position (absolute) control**; min-max action normalization is "critical"; velocity-style control *handicaps* diffusion policy (about −0.2 on Square) while *helping* BC.
  - Our runs used robomimic's **default** delta actions with no action normalization. That's the shipped template, so it's not a deviation from robomimic, but it differs from Chi's published setup. can/mg actions already span [−1,1].
  - So action parameterization is the leading candidate explanation. **The papers say the policy-class reading is provisional.**
- **D3, deferred until workshop notifications:** can/ph with three arms: (1) observation normalization off, (2) `hdf5_normalize_obs` on, (3) absolute actions plus min-max normalization. Arm 3 is the only one comparable to Chi's 0.96. can/ph isn't downloaded yet. Conversion scripts exist (`robosuite_add_absolute_actions.py`, `extract_action_dict.py`). Compare our late-checkpoint mean against their last-10 average.

## Papers (all in paper/; each edited separately; submissions/ = frozen live copies)
| File | Venue | Limit | Status |
|---|---|---|---|
| paper_oopsie.tex | CoRL 2026 "Oops, I Erred" workshop | 8 pp, excl. refs and appendix | **Live and final.** OpenReview 7C6WROjCKQ; PDF sha1 ab2a0e17, verified by download. Zero spare lines |
| paper_4page.tex | CoRL 2026 WEBP ("Everything Beneath the Policy") | 4 pp, excl. refs | **Live and final.** 8HHUxS9GcK; sha1 9ca67e37, verified. Zero spare lines. Notification Oct 26 |
| paper_robopad.tex | NeurIPS 2026 RoboPAD workshop | 9 pp main content | Window closed Sep 13; live = Sep 8 original (BC only). Working copy updated but about a page over, for any camera-ready |
| paper.tex | arXiv / ResearchGate, not anonymized | none | Updated, not yet posted |

- All three workshops are **non-archival** (WEBP confirmed by organizer Younghyo Park, Sep 24). WEBP requires disclosing concurrent submissions before Oct 26.
- Notifications run Sep 29 through Oct 26.
- **Wording conventions in all four:**
  - TED harm is established **under BC only**.
  - The jerk result is **"in the configuration we tested"**.
  - The metric-level reading of TED is **"consistent with"**, never "supports".
  - Say "policy **configuration** (architecture and action parameterization)", not "policy class".
  - Limitations name **REASSEMBLE/DROID**; LIBERO has no failures.

## Next steps
1. After notifications: D3 (the three arms above).
2. Real human-failure extension, **detection only** (no simulator):
   - **REASSEMBLE first:** 516 failed segments of 4,551; Franka, end-effector xyz+quat; CC BY 4.0; 54.8 GB zip with video included. **Check disk space first.** About 1–1.5 weeks.
   - **Then DROID:** about 16k operator-labeled failures in `failure/` folders; end-effector pose and gripper at 15 Hz; CC BY 4.0. Trajectory files are about 1.35 MB per episode, roughly 20 GB for all failures with video excluded. About 2–3 weeks.
   - **Skip RH20T** (mixed rigs and rates).
   - Expect failure episodes to be shorter, so report length-alone AUROC first.
3. Archival target: IROS 2027, RA-L, or CoRL 2027 (about May, 8 pp). Needs D3, likely the failure-pool result, and an IEEE two-column rewrite (on hold).
4. Make the GitHub repo **public again before posting to arXiv**. It went private Sep 23 for double-blind review.

## Repo / environment
- Repo: github.com/imjbassi/smoothness-metric-audit (private). Commits must be `imjbassi <jaiveerbassi@yahoo.com>`, **with no AI co-author lines**.
- Windows local: `C:\Users\jaive.DESKTOP-3TNM9JL\Desktop\smoothness-metric-audit`.
- Training: WSL `~/robomimic`, `.venv` (torch 2.13 cu130, robomimic 0.5.0, robosuite 1.4.1, mujoco 2.3.2, diffusers 0.11.1, RTX 4070).
- Raw runs: `results_train/` (70 BC) and `results_train_dp/` (25 diffusion). Gitignored. Configs are in `configs/` and `configs_dp/`.
- `python results/collect2.py --from-csv results/train_results_all.csv` reproduces every reported number.
- Build: run pdflatex from `paper/` (figures resolve via `\graphicspath{{../figures/}}`). PDFs are tracked in git; rebuild after .tex changes.
- **Always verify an OpenReview upload** by downloading the PDF and comparing SHA1. A WEBP re-upload once silently failed.
