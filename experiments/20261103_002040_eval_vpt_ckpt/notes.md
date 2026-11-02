# 100k mask warm-start — easy AI (20 seeds)

Checkpoint: `experiments/flat_100k_mask_warm/checkpoints/2026-11-02_23-43-23/2/`  
Config: `configs/env/zombie_stage1_easy.yaml` (front spawn, AI zombie)  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.35 death_rate=0.50 timeout_rate=0.15
mean_hp_lost=13.90 mean_steps=309.50
mean_reward=+1.54 mean_time_to_kill=221.14
```

The policy transfers to a counterattacking zombie in front: 35% win, but 50%
death. Defense is still weak.
