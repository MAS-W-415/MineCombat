# M3 — Angle Curriculum

Date: 2026-11-03  
Base checkpoint: `experiments/probe_easy_far/checkpoints/2026-11-03_01-18-07/20/`  
Training config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Checkpoint: `experiments/flat_100k_angle30_mask_warm/checkpoints/2026-11-03_02-11-52/2/`

## Why

The far-spawn checkpoint reached 52% win / 40% deaths on the easy front task,
but a ±30° spawn offset dropped it to 10% win / 80% deaths. Distance generalized
(40% on mixed 3–7); angle did not. This run warm-started 100k on ±30° jitter.

## Results

| Eval task | Before (far) | After (100k angle) | Delta |
|---|---:|---:|---:|
| easy 3–4, front (50 seeds) | 0.52 | 0.44 | −0.08 |
| ±30°, 3–4 (50 seeds) | 0.10 | 0.16 | +0.06 |
| mixed 3–7, front (20 seeds) | 0.40 | 0.30 | −0.10 |
| full random (20 seeds) | 0.10 | 0.10 | 0.00 |

After the angle run:

- easy: 56% deaths, HP lost 14.18.
- ±30°: 78% deaths; 21 of 39 deaths dealt zero damage.
- random: deaths fell from 75% to 40%, but timeouts rose to 50%. All 10 timeout
  episodes dealt zero damage, so the policy learned avoidance, not engagement.

## Interpretation

The 100k run did not produce a clear gain. The ±30° target improved by only 6
points, while easy and mixed regressed by 8–10 points. Fifty seeds have a
standard error of roughly 7 points for a 0.5 win rate, so the angle gain is
within noise; the easy/mixed regressions are also near noise but point in the
wrong direction.

Two facts argue that 100k at one environment is too small for a new behavior:

1. The easy-task gain from far training appeared after 10k, but it reused the
   existing approach/attack behavior.
2. Off-angle combat requires a new camera-control policy. The 100k run collected
   only ~200 episodes, and most off-angle episodes still ended without a single
   hit.

## Next

A joint 100k run on mixed distance 3–7 + ±30° jitter is running from the same
far checkpoint. If it also fails to improve angle transfer, move the next run
to the 1M host instead of stacking more 100k variants. The 1M curriculum should
use the mixed + angle config and multiple environments.

## Artifacts

- `experiments/20261103_024205_eval_vpt_ckpt/` (easy, 50 seeds)
- `experiments/20261103_024757_eval_vpt_ckpt/` (±30°, 50 seeds)
- `experiments/20261103_025046_eval_vpt_ckpt/` (mixed, 20 seeds)
- `experiments/20261103_025404_eval_vpt_ckpt/` (random, 20 seeds)
