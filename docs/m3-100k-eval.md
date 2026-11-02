# M3 — 100k Flat PPO Checkpoint Evaluation

Date: 2026-11-02  
Checkpoint: `experiments/flat_100k/checkpoints/2026-11-02_16-10-12/2/model.ckpt`  
Training: 2 iterations × 200 batches × 256 steps = 102,400 steps  
Seeds: 0–49, max 600 steps, deterministic

## Result

| Metric | Zero-shot VPT | 10k dense | 100k flat PPO |
|---|---:|---:|---:|
| Win rate | 0.02 | 0.00 | 0.00 |
| Death rate | 0.28 | 0.36 | 0.16 |
| Timeout / stuck rate | 0.70 | 0.64 | 0.84 |
| Mean HP lost | 11.58 | 10.92 | 10.28 |
| Mean steps | 535.7 | 527.3 | 579.2 |
| Damage ratio | 0.00* | 0.0090 | 0.0313 |

\* The zero-shot and 10k runs predate the damage-stat unit fix; their damage
ratio is zero or was corrected post-hoc. The 10k value here is the corrected
HP-based ratio.

Per-episode behavior:

- Only **4 of 50** episodes dealt any damage.
- Maximum damage in an episode: **4.9 HP** (a zombie has 20 HP).
- No episode killed the zombie.
- 84% of episodes hit the 600-step timeout.

Training wall time: 24.8 minutes.

## Interpretation

The 100k checkpoint is less likely to die and loses slightly less HP than
zero-shot, but it is also more likely to time out and still never wins. The
damage ratio increased from 0.0090 at 10k to 0.0313, so there is a weak signal
that the policy occasionally attacks. It is not a usable combat policy.

The training video shows the agent wandering, looking at terrain, and opening
the inventory; it does not show purposeful zombie approach and attack. This is
consistent with the per-episode damage distribution.

## Evaluation noise

Two independent 50-seed evaluations of the same checkpoint gave noticeably
different death rates (30% vs 16%) and timeout rates (70% vs 84%). Minecraft
physics, zombie AI, and reset details are not fully deterministic even with
seeded spawn/yaw. Treat 50-episode point estimates as noisy; the stage-1 gate
should use more episodes or confidence intervals.

## Next

Do not start the 1M run blindly. The 100k result suggests the policy may be
learning to avoid damage rather than to kill. Run one targeted ablation first:

1. Mask or penalize inventory/GUI actions.
2. Rebalance the reward: reduce the HP-loss penalty or increase damage/kill
   reward to encourage engagement.
3. Try a closer/in-front zombie spawn curriculum for early training.

Then run another 100k and compare. If damage and win rate move in the right
direction, spend the GPU time on 1M.

## Artifacts

- `experiments/20261102_165805_eval_vpt_ckpt/`
- Training videos: `output/videos/2026-11-02_16-08-46/`
