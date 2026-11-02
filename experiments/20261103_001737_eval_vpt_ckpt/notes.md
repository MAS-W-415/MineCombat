# 100k mask warm-start — stationary (50 seeds)

Checkpoint: `experiments/flat_100k_mask_warm/checkpoints/2026-11-02_23-43-23/2/`  
Config: `configs/env/zombie_stage1_stationary.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.22 death_rate=0.00 timeout_rate=0.78 stuck_rate=0.34
mean_hp_lost=0.00 mean_steps=515.46
mean_reward=+2.50 mean_time_to_kill=215.73
```

22% win rate with zero deaths. This is the first stable combat signal in the
project.
