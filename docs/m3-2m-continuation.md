# M3 — 2M Continuation of Flat PPO

Date: 2026-11-06  
Source checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Hardware: NVIDIA RTX 4090 D (24 GB), 60 GB RAM  
Code: `b54c53d`  
Environment: Python 3.10.22, PyTorch 2.5.1+cu124, MineStudio 1.1.6

## Training

A second 1,024,000-step run, warm-started from the 1M checkpoint. This brings
the total to 2.048M steps.

```text
environment steps: 1,024,000
PPO iterations: 20
wall time: 5,603.46 s (93.4 min)
peak GPU memory: 6,039 MiB
peak system RAM: 39 GB
NaN / Inf: 0
too-high-KL warnings: 4,572
```

`--keep-interval 5` retained all four checkpoints:

| Name | Total steps | Path |
|---|---:|---|
| iteration 5 | 1.25M | `checkpoints/2026-11-06_11-45-28/5/` |
| iteration 10 | 1.50M | `checkpoints/2026-11-06_11-45-28/10/` |
| iteration 15 | 1.75M | `checkpoints/2026-11-06_11-45-28/15/` |
| iteration 20 | 2.00M | `checkpoints/2026-11-06_11-45-28/20/` |

W&B was run offline. The offline `.wandb` file is preserved, but upload failed
with HTTP 403 because no API key was configured, so there is no public W&B
curve for this run.

## Intermediate trend (local, single 1×50)

These are local RTX 4060 evaluations with hardware rendering, used only for the
trend. The final 2M gate evaluation is on the RTX 4090D.

| Total steps | Combined 3–7 + ±30° | Random 360° |
|---|---:|---:|
| 1.0M (remote 3×50 mean) | 0.407 | 0.227 |
| 1.25M | 0.56 | — |
| 1.50M | **0.68** | — |
| 1.75M | 0.64 | 0.20 |

The combined task rose from 40.7% at 1M to 56–68% at 1.25–1.75M. Random remains
around 20%. This suggests the visible-combat gate can pass in the 1.5–2M range,
while full 360° search remains the separate bottleneck.

## Final 2M evaluation (remote 3×50)

| Task | Win | Death | HP lost | Time-to-kill |
|---|---:|---:|---:|---:|
| easy 3–4 | **0.90** | 0.10 | 6.96 | 166.7 |
| ±30° close | **0.64** | 0.30 | 11.34 | 189.0 |
| combined 3–7 + ±30° r1 | 0.58 | 0.38 | 12.42 | 211.5 |
| combined r2 | 0.66 | 0.32 | 12.35 | 220.4 |
| combined r3 | 0.62 | 0.32 | 11.68 | 204.3 |
| **combined mean** | **0.620** | **0.340** | **12.15** | **212.1** |
| random 360° r1 | 0.06 | 0.54 | 14.72 | 288.3 |
| random r2 | 0.10 | 0.60 | 15.00 | 270.0 |
| random r3 | 0.14 | 0.54 | 15.08 | 270.0 |
| **random mean** | **0.100** | **0.560** | **14.93** | **276.1** |

Combined range: **0.58–0.62**. Random range: **0.06–0.14**.

### 1M → 2M comparison

| Task | 1M | 2M | Delta |
|---|---:|---:|---:|
| easy | 0.72 | 0.90 | +0.18 |
| ±30° | 0.52 | 0.64 | +0.12 |
| combined | 0.407 | 0.620 | +0.213 |
| random | 0.227 | 0.100 | **−0.127** |

## Decision

The result splits cleanly:

- **Stage-1a — visible combat: passes.** Combined mean is 62% (range 58–62%),
  easy is 90%, and ±30° is 64%. Video review shows real approach, aim, attack,
  and kill.
- **Stage-1b — 360° search: fails.** Random mean is 10% (range 6–14%), down
  from 22.7% at 1M. Continuing the ±30° curriculum improves visible combat but
  degrades search.

Do **not** run another 1M on the same ±30° curriculum expecting random to
improve. The next work should be a search-specific curriculum or mechanism
(scan-at-reset, auto-face, or a relative-direction observation for HH), and the
1M checkpoint is a better starting point for search than the 2M checkpoint.

For visible-combat work (including a future HH1 initialization), the 2M
checkpoint is the strongest flat policy. The Stage-1 gate is therefore recorded
as **Stage-1a passed, Stage-1b open**, not as a single full-random pass.

## Artifacts

- Training logs: `experiments/flat_1m_cont_mixed_angle30_4090d/`
- Intermediate local evals: `experiments/eval_cont_intermediate_local/`
- Final remote evals: `experiments/eval_2m_4090d/`
- Offline W&B: `experiments/wandb_offline_2m/` (local, not synced)
- Checkpoints: host + group share (`minecombat_results/flat_2m_mixed_angle30_4090d/`)
