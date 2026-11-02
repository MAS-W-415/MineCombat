# M3 — Defense Probes After the Easy-AI Plateau

Date: 2026-11-03  
Source checkpoint: `experiments/flat_100k_easy_mask_warm/checkpoints/2026-11-03_00-33-34/2/`  
Probe length: 10,240 steps (warm-started), 1 environment

## Why

The 100k easy-AI run plateaued at 30% win with 58% deaths. Two cheap probes
tested whether the bottleneck was reward weighting or engagement distance:

| Probe | Change |
|---|---|
| easy_control | continue the same easy-AI task |
| easy_defense | `damage_taken` 0.2 → 0.3 |
| easy_far | front spawn 3–4 → 5–7 blocks |

## 20-seed screen

| Probe | Eval task | Win | Death | HP lost |
|---|---|---:|---:|---:|
| easy_control | easy | 0.20 | 0.65 | 14.4 |
| easy_defense | easy | 0.40 | 0.60 | 14.6 |
| easy_far | easy | 0.40 | 0.55 | 11.6 |
| easy_far | far | 0.50 | 0.40 | 13.0 |

The 20-seed screen was misleading: `easy_far` looked best on its own far task
(50%), and the other two probes looked similar to the control.

## 50-seed confirmation

| Probe | Eval task | Win | Death | HP lost | HP lost on win | Time-to-kill |
|---|---|---:|---:|---:|---:|---:|
| easy_control | easy | 0.26 | 0.62 | 15.20 | 9.69 | 195.8 |
| easy_defense | easy | 0.26 | 0.68 | 15.96 | 11.08 | 214.2 |
| **easy_far** | **easy** | **0.52** | **0.40** | **11.64** | **6.81** | **184.0** |
| easy_far | far | 0.32 | 0.54 | 14.12 | 7.69 | 215.3 |

`easy_far` is the winner: 52% win and 40% deaths on the close-range easy task,
with a mean reward of +3.89. The 20-seed far result (50%) did not hold; the
far task itself is only 32% win at 50 seeds.

## Interpretation

1. **Stronger HP-loss reward is not the answer.** The defense probe matched the
   control on wins and was worse on deaths. Its attack-button rate rose to 89%,
   so it became more aggressive instead of learning to retreat.
2. **Engagement is the bottleneck.** In the 100k easy checkpoint, 30/50
   episodes dealt zero damage but 29 episodes still ended in death. The policy
   often fails to acquire or reach the zombie, then gets killed while passive.
3. **Far training transfers to close range.** Training at 5–7 blocks improved
   the 3–4 block task from 26% to 52% wins and cut HP lost per win from 9.7 to
   6.8. The likely mechanism is that the zombie must walk toward the player, so
   the policy gets a clean approach phase and learns to time attacks as the
   target closes.
4. **20-seed screens are only for triage.** Two of the four 50-seed results
   moved by more than 15 percentage points from the 20-seed screen.

## Zero-shot transfer of the far checkpoint (20 seeds)

| Eval task | Win | Death | HP lost | HP lost on win |
|---|---:|---:|---:|---:|
| easy 3–4, front | 0.52 (50 seeds) | 0.40 | 11.64 | 6.81 |
| ±30° jitter, 3–4 | 0.10 | 0.80 | 17.70 | 15.00 |
| mixed distance 3–7, front | 0.40 | 0.50 | 11.40 | 3.75 |
| full random spawn | 0.10 | 0.75 | 17.00 | 18.00 |

Distance generalizes: mixed 3–7 still holds 40% wins, and close-range wins take
only 3.8 HP. Angle does not: a ±30° offset drops the win rate to 10% and raises
deaths to 80%. Random spawn produced the first 2/20 wins, both at 18 HP lost.

The next experiment is therefore 100k warm-started on the ±30° config, not a
distance change.

## Next

Use `experiments/probe_easy_far/checkpoints/2026-11-03_01-18-07/20/` as the new
base checkpoint:

1. Warm-start 100k on the ±30° spawn-jitter curriculum.
2. Evaluate on easy front (50 seeds), ±30° (50 seeds), and random spawn
   (20–50 seeds). Keep it only if easy-front win rate stays at or above 40%;
   otherwise the angle step is too aggressive.
3. Widen to ±60° and then 360°, mixing distance back in once angle transfer
   holds.

The far task itself still needs work (32% win), but it is no longer the target:
the goal is random spawn. Distance mixing should preserve the close-range gain
while keeping the approach behavior learned at 5–7 blocks.

## Artifacts

- `experiments/probe_easy_control/`
- `experiments/probe_easy_defense/`
- `experiments/probe_easy_far/`
- `experiments/20261103_014156_eval_vpt_ckpt/` (control, 50 seeds)
- `experiments/20261103_014746_eval_vpt_ckpt/` (defense, 50 seeds)
- `experiments/20261103_015319_eval_vpt_ckpt/` (far → easy, 50 seeds)
- `experiments/20261103_015928_eval_vpt_ckpt/` (far → far, 50 seeds)
- `experiments/20261103_020329_eval_vpt_ckpt/` (far → ±30°, 20 seeds)
- `experiments/20261103_020610_eval_vpt_ckpt/` (far → mixed 3–7, 20 seeds)
- `experiments/20261103_020910_eval_vpt_ckpt/` (far → random, 20 seeds)
