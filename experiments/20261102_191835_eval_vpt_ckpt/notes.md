# 100k stationary-target checkpoint — stationary environment

Checkpoint: `experiments/flat_100k_stationary/checkpoints/2026-11-02_18-38-41/2/model.ckpt`  
Evaluation config: `configs/env/zombie_stage1_stationary.yaml` (NoAI zombie, 2.5–3.5 blocks in front)  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.00 timeout_rate=1.00 stuck_rate=0.96
mean_hp_lost=0.08 mean_steps=600.00 mean_reward=-0.59
damage_ratio=2.95
```

The high damage ratio is misleading: total HP loss is almost zero, so the
denominator is tiny. The policy dealt very little damage overall (mean ~0.24 HP
per episode) and never killed the stationary zombie.

Action diagnostic on the same checkpoint:

```text
1,200 steps: 10 attack presses (0.8%)
```

A separate 1-episode run produced 0 attack presses in 600 steps. The policy
mostly rotates the camera and strafes; it does not press attack.

Conclusion: the stationary target did not teach attack. The bottleneck is
action discovery, not target visibility or reward magnitude.
