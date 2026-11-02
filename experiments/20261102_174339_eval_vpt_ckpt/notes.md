# 100k R1 reward — 50-seed evaluation

Checkpoint: `experiments/flat_100k_r1/checkpoints/2026-11-02_17-11-28/2/model.ckpt`  
Reward: R1 aggression profile  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.36 timeout_rate=0.64 stuck_rate=0.60
mean_hp_lost=11.68 mean_steps=541.08 mean_reward=-2.86
damage_ratio=0.0168
```

Only 2/50 episodes dealt damage; max 4.9 HP; no kills.

Compared with the baseline 100k checkpoint:

| Metric | Baseline 100k | R1 |
|---|---:|---:|
| Win rate | 0.00 | 0.00 |
| Death rate | 0.16 | 0.36 |
| Timeout rate | 0.84 | 0.64 |
| Mean HP lost | 10.28 | 11.68 |
| Damage ratio | 0.0313 | 0.0168 |
| Episodes with damage | 4/50 | 2/50 |

R1 did not improve engagement. It reduced timeout and increased deaths, but did
not increase damage. The training video still shows wandering/terrain viewing,
not a zombie approach.
