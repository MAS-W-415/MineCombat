# 100k easy curriculum — 50-seed evaluation

Checkpoint: `experiments/flat_100k_easy/checkpoints/2026-11-02_17-51-57/2/model.ckpt`  
Curriculum: zombie 3–4 blocks directly in front, player yaw 0  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.02 death_rate=0.32 timeout_rate=0.66 stuck_rate=0.64
mean_hp_lost=11.80 mean_steps=541.58 mean_reward=-2.65
damage_ratio=0.0422
```

Per-episode:

- 2/50 episodes dealt damage.
- 1/50 killed the zombie (seed 47, 20 HP damage, time-to-kill 356).
- The other damage episode dealt 4.9 HP.
- The remaining 48 episodes dealt no damage.

The training video shows the zombie clearly in front and often very close, but
the agent does not attack consistently; it is frequently killed at close range.
