# Phase 1 probe evaluation — inventory mask (50 seeds)

Checkpoint: `experiments/probe_mask/checkpoints/2026-11-02_23-10-56/20/model.ckpt`  
Config: `configs/env/zombie_stage1_stationary.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.08 death_rate=0.00 timeout_rate=0.92 stuck_rate=0.50
mean_hp_lost=0.20 mean_steps=570.42
mean_reward=+0.748 mean_time_to_kill=230.25
```

The 20-seed result held up: 4/50 kills, zero deaths, positive mean reward for
the first time. This is the Phase 1 winner and the warm-start source for the
100k continuation.
