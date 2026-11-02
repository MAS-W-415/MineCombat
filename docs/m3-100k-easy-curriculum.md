# M3 — 100k Easy Curriculum

Date: 2026-11-02  
Config: `configs/env/zombie_stage1_easy.yaml`  
Checkpoint: `experiments/flat_100k_easy/checkpoints/2026-11-02_17-51-57/2/model.ckpt`  
Seeds: 0–49, max 600 steps, deterministic

## Motivation

The baseline 100k policy mostly wandered; the training video rarely showed the
zombie. This curriculum removes the search problem:

- zombie at 3–4 blocks, directly in front (`spawn_angle_deg=90`)
- player yaw fixed at 0
- baseline reward (not R1)

The question: can PPO learn to approach and attack when the target is already
visible?

## Result

| Metric | Baseline 100k | R1 100k | Easy 100k |
|---|---:|---:|---:|
| Win rate | 0.00 | 0.00 | **0.02** |
| Death rate | 0.16 | 0.36 | 0.32 |
| Timeout rate | 0.84 | 0.64 | 0.66 |
| Mean HP lost | 10.28 | 11.68 | 11.80 |
| Mean steps | 579.18 | 541.08 | 541.58 |
| Damage ratio | 0.0313 | 0.0168 | **0.0422** |
| Episodes with damage | 4/50 | 2/50 | 2/50 |
| Max damage | 4.9 HP | 4.9 HP | **20.0 HP** |

Per-episode:

- One full kill: seed 47, 20 HP damage, time-to-kill 356.
- One partial hit: 4.9 HP.
- The other 48 episodes dealt no damage.

Training wall time: 25.5 minutes.

## Interpretation

The easy curriculum produced the first kill and the highest damage ratio, but
only 2/50 episodes dealt any damage. The policy can attack when everything is
aligned, but it is not reliable.

The video shows the zombie clearly in view and often very close, yet the agent
does not attack consistently; it is frequently killed at close range. So the
remaining problem is not "finding the enemy". It is the attack/aim/timing
behavior itself.

## Next

The cheapest next step is a stationary-target curriculum:

1. Spawn a **NoAI** zombie 2–3 blocks in front.
2. Let the agent learn approach + attack + kill without being killed first.
3. Then enable zombie AI (the current easy curriculum).
4. Then restore random ring spawn.

Also consider a 300k run on the easy curriculum now that a positive kill sample
exists. If 300k does not raise the win rate, the stationary-target stage is the
better next move.

Artifacts:

- `experiments/20261102_182349_eval_vpt_ckpt/`
- Training video: `output/videos/2026-11-02_17-50-31/51200 - ...mp4`
