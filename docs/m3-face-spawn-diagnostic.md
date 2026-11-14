# M3 — Face-Spawn Diagnostic

Date: 2026-11-14  
Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Task: random distance 4–10 + 360° spawn  
Protocol: 50 seeds, max 600 steps, deterministic, local RTX 4060

## Why

The 2M checkpoint scores 62% on the combined ±30° task but only 10% on random
spawn. The `face_spawn` option turns the player toward the first zombie at
reset, so the zombie is initially visible while distance and world orientation
stay random. This separates **target acquisition/search** from **combat after
acquisition**.

## Results

| Task | Win | Death | Timeout | Mean damage | Episodes with damage |
|---|---:|---:|---:|---:|---:|
| random_face | **0.36** | 0.48 | 0.16 | **8.9 HP** | **52%** |
| random | 0.10 | 0.46 | 0.44 | 2.5 HP | 16% |

Facing the target at reset triples engagement:

- win rate 0.10 → 0.36
- mean damage 2.5 → 8.9 HP
- episodes with any damage 16% → 52%
- timeout rate 44% → 16%

## Distance breakdown

### random_face

| Distance | Win | Death | Episodes with damage |
|---|---:|---:|---:|
| 4–5 | 0.31 | 0.54 | 46% |
| 5–6 | 0.22 | 0.56 | 33% |
| 6–7 | 0.17 | 0.83 | 33% |
| 7–8 | 0.67 | 0.33 | 78% |
| 8–10 | 0.38 | 0.31 | 62% |

### random

| Distance | Win | Death | Episodes with damage |
|---|---:|---:|---:|
| 4–5 | 0.15 | 0.46 | 23% |
| 5–6 | 0.11 | 0.78 | 22% |
| 6–7 | 0.17 | 0.33 | 17% |
| 7–8 | 0.00 | 0.22 | 0% |
| 8–10 | 0.08 | 0.46 | 15% |

## Interpretation

1. **Acquisition is a major bottleneck.** Simply facing the zombie at reset
   raises engagement from 16% to 52% of episodes and triples mean damage.
2. **Acquisition is not the only bottleneck.** Even when facing, the policy
   still dies in 48% of episodes and wins only 36%. Close/mid distances (4–7)
   are worse than 7–10, suggesting the policy loses the target during the
   fight or fails to react when the zombie is already close.
3. **Stage-1b has two subproblems**: initial search/acquisition and in-combat
   camera tracking. A scan-at-reset or auto-face curriculum addresses the
   first; tracking remains a separate issue.

## Next

1. Record 5–10 `random_face` episodes locally and inspect whether the zombie is
   actually in view at reset and how the camera loses it.
2. Implement scan-at-reset and compare it with spawn-facing on the same 50
   seeds.
3. If facing + scan still leaves close-range deaths high, add a tracking /
   close-range curriculum rather than more random search training.
4. Keep 2M as the visible-combat checkpoint; do not continue the ±30° run.

## Artifacts

- `experiments/eval_2m_face_local/20261114_131639_eval_vpt_ckpt/` (face)
- `experiments/eval_2m_face_local/20261114_132359_eval_vpt_ckpt/` (random)
