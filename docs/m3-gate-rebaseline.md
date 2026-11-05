# M3 — Stage-1 3×50 Re-Baseline

Date: 2026-11-05  
Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Protocol: 3 independent evaluations × 50 seeds (0–49), max 600 steps, deterministic

## Why

Single 50-seed runs of the same checkpoint varied by up to 14 points because
Minecraft mob AI is not reset between episodes. The Stage-1 decision needs an
interval, not a single number.

## Results

### Combined task: distance 3–7 + ±30°

Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`

| Rep | Win | Death | Timeout | HP lost | Steps |
|---|---:|---:|---:|---:|---:|
| 1 | 0.22 | 0.62 | 0.16 | 15.02 | 330.4 |
| 2 | 0.10 | 0.64 | 0.26 | 15.48 | 365.5 |
| 3 | 0.16 | 0.58 | 0.26 | 15.60 | 358.2 |
| **Mean** | **0.16** | **0.61** | **0.23** | **15.37** | **351.4** |

Range: 0.10–0.22.

### Random 360°

Config: `configs/env/zombie_stage1.yaml`

| Rep | Win | Death | Timeout | HP lost | Steps |
|---|---:|---:|---:|---:|---:|
| 1 | 0.08 | 0.56 | 0.36 | 14.84 | 433.1 |
| 2 | 0.08 | 0.76 | 0.16 | 16.22 | 387.0 |
| 3 | 0.14 | 0.58 | 0.28 | 15.58 | 414.9 |
| **Mean** | **0.10** | **0.63** | **0.27** | **15.55** | **411.7** |

Range: 0.08–0.14.

## Interpretation

1. **Visible-combat gate still fails.** The combined task is 16% mean, 34 points
   below the 50% gate. The 100k curriculum has plateaued at this checkpoint.
2. **Random 360° remains a search problem.** 10% mean is better than the earlier
   4% single run, but the interval is 8–14%; it is not close to the gate.
3. **The next experiment is 1M, not another 100k.** The 100k checkpoint has been
   re-baselined; more 100k variants would be measured against a 10-point noise
   floor. A 1M run on the combined task is the remaining planned test of
   whether scale closes the gap.
4. **Do not claim Stage-1a or Stage-1.** Both gates remain open.

## Next

Run the 1M handoff on the RTX 4090D (24 GB / 60 GB / 24 h):

- Base: this checkpoint.
- Curriculum: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`.
- 20 iterations × 51,200 steps = 1,024,000 steps.
- Save every 5 iterations (250k, 500k, 750k, 1M).
- Evaluate 250k/500k with 1×50 and the final checkpoint with 3×50 on every
  task.
- See `docs/1m-4090d-handoff.md`.

If the 1M combined task remains below ~30%, the visible-combat gate is not
achievable with flat VPT + PPO at this scale; the next step is a curriculum or
task-definition change, not another long run.

## Artifacts

- `experiments/20261105_190512_eval_vpt_ckpt/` (combined rep1)
- `experiments/20261105_191839_eval_vpt_ckpt/` (combined rep2)
- `experiments/20261105_193153_eval_vpt_ckpt/` (combined rep3)
- `experiments/20261105_191212_eval_vpt_ckpt/` (random rep1)
- `experiments/20261105_192522_eval_vpt_ckpt/` (random rep2)
- `experiments/20261105_193850_eval_vpt_ckpt/` (random rep3)
