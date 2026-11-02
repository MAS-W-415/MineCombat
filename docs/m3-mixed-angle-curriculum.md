# M3 — Mixed Distance + Angle Curriculum

Date: 2026-11-03  
Base checkpoint: `experiments/probe_easy_far/checkpoints/2026-11-03_01-18-07/20/`  
Training config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`

## Why

Two single-axis probes left the policy stuck:

- 100k on ±30° alone reached only 16% on ±30° and regressed easy to 44%.
- The far checkpoint held 40% zero-shot on mixed distance but only 10% on ±30°.

The joint curriculum trains distance 3–7 and ±30° jitter together, so the policy
sees both the approach phase and an off-center target.

## Results

### 50-seed confirmations

| Eval task | Far base | Joint 100k |
|---|---:|---:|
| easy 3–4, front | 0.52 | 0.44 |
| ±30° jitter | 0.10 | **0.34** |
| mixed 3–7, front | 0.40 (20 seeds) | **0.48** |
| full random | 0.10 (20 seeds) | 0.04 |

Joint 100k detail:

| Task | Win | Death | Timeout | HP lost | HP lost on win | Time-to-kill |
|---|---:|---:|---:|---:|---:|---:|
| easy | 0.44 | 0.48 | 0.08 | 12.38 | 6.68 | 192.1 |
| ±30° | 0.34 | 0.52 | 0.14 | 13.66 | 7.76 | 198.5 |
| mixed 3–7 | 0.48 | 0.48 | 0.04 | 13.38 | 9.00 | 211.7 |
| random | 0.04 | 0.64 | 0.32 | 15.10 | 9.00 | 208.0 |

## Interpretation

1. **The joint curriculum works on its trained distribution.** ±30° rose from
   10% to 34%, and mixed distance settled at 48%. Both are close to or above the
   50% gate for their respective tasks.
2. **It does not generalize to 360° yet.** Random spawn fell to 2/50 wins (4%).
   The policy learned to handle targets near the front, not targets behind it.
3. **Twenty seeds cannot resolve a low-probability effect.** The random-spawn
   screen showed 15% (3/20); the 50-seed result was 4% (2/50). From now on,
   random-spawn claims require 50 seeds, and any 20-seed screen is labeled as
   triage only.
4. **Close-range retention costs a little.** Easy dropped from 52% to 44% and
   deaths rose from 40% to 48%, but this is within about one standard error and
   is an acceptable trade for the angle gain.

## Step 2: widen to ±60°

100k warm-started from the joint ±30 checkpoint on
`configs/env/zombie_stage1_easy_mixed_angle60_mask.yaml`.

| Eval task | ±30 checkpoint | ±60 checkpoint |
|---|---:|---:|
| easy 3–4, front | 0.44 | **0.58** |
| ±60° jitter | — | **0.08** |
| mixed 3–7 | 0.48 | 0.45 (20 seeds) |
| full random | 0.04 | 0.04 |

The wider run improved easy front to 58% but collapsed on its own target: 8%
win and 74% deaths at ±60°. At ±60° many spawns are outside the field of view,
so the zombie is not visible at reset. The policy appears to learn a local
optimum: wait for the zombie to enter the view instead of turning to search.
This is the off-screen search problem, not a distance or defense problem.

Widening further to ±90° or ±180° without an exploration change is unlikely to
help.

## Camera entropy A/B (rejected)

Same base checkpoint and ±60° environment, with
`--entropy-coef 0.01 --camera-entropy-scale 5`.

| Eval task | ±60 baseline | + camera entropy |
|---|---:|---:|
| easy 3–4, front | 0.58 | 0.36 |
| ±60° jitter | 0.08 | 0.10 |
| mixed 3–7 | 0.45 (20 seeds) | 0.45 (20 seeds) |
| full random | 0.04 | 0.02 |

The ±60° target moved by 2 points (within noise), while easy fell by 22 points
and random fell by 2. Camera entropy at this scale disrupts front-facing combat
more than it helps search. Reject this direction.

## The search problem

The zombie walks straight toward the player, so its bearing stays constant as it
approaches. A zombie spawned at ±60° never enters the field of view on its own.
The policy must turn to find it, but the image observation contains no
directional cue while the target is off-screen. This is a search problem, not an
aiming or defense problem.

Flat VPT + PPO with image-only observations has no memory of where an unseen
target is. Options:

1. **Split the gate**: Stage-1a requires ≥50% on visible random spawn (distance
   3–7, angle within ±30–35°); Stage-1b treats 360° search separately.
2. **Add a cue or assist**: auto-face on spawn, a scan-at-reset curriculum, or a
   relative-direction observation (the last is HH territory).
3. **Invest in search learning**: 1M+ steps with stronger exploration or a
   recurrent search policy. High risk for flat PPO.

## Next

Update: the joint ±30 checkpoint was evaluated on the combined config
(`configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`) for 50 seeds three
times: 10%, 24%, and 20% (mean ~18%). Composition is weak even though the
separate tasks look much better. See `docs/m3-evaluation-variance.md` for the
variance analysis.

Decision point:

- For a **visible-combat gate**, re-baseline the combined config with 3×50
  seeds and decide whether to keep training that curriculum (e.g. a 1M run on
  the host) or simplify the gate to a single visible task.
- For a **full 360° gate**, add a search mechanism: scan-at-reset or auto-face
  for curriculum, or a relative-direction observation for HH. More flat 100k
  width stages are not justified by the current evidence.

## Artifacts

- `experiments/20261103_035419_eval_vpt_ckpt/` (easy, 50 seeds)
- `experiments/20261103_040009_eval_vpt_ckpt/` (±30°, 50 seeds)
- `experiments/20261103_041145_eval_vpt_ckpt/` (mixed, 50 seeds)
- `experiments/20261103_041850_eval_vpt_ckpt/` (random, 50 seeds)
- `experiments/20261103_040252_eval_vpt_ckpt/` (mixed, 20-seed screen)
- `experiments/20261103_040555_eval_vpt_ckpt/` (random, 20-seed screen)
- `experiments/20261103_045146_eval_vpt_ckpt/` (easy, ±60 checkpoint, 50 seeds)
- `experiments/20261103_045802_eval_vpt_ckpt/` (±60, ±60 checkpoint, 50 seeds)
- `experiments/20261103_050045_eval_vpt_ckpt/` (mixed, ±60 checkpoint, 20 seeds)
- `experiments/20261103_050733_eval_vpt_ckpt/` (random, ±60 checkpoint, 50 seeds)
- `experiments/20261103_053947_eval_vpt_ckpt/` (easy, camera entropy, 50 seeds)
- `experiments/20261103_054618_eval_vpt_ckpt/` (±60, camera entropy, 50 seeds)
- `experiments/20261103_054908_eval_vpt_ckpt/` (mixed, camera entropy, 20 seeds)
- `experiments/20261103_055552_eval_vpt_ckpt/` (random, camera entropy, 50 seeds)
