# 100k mask warm-start — easy AI (50 seeds)

Checkpoint: `experiments/flat_100k_mask_warm/checkpoints/2026-11-02_23-43-23/2/`  
Config: `configs/env/zombie_stage1_easy.yaml` (front spawn, AI zombie)  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.30 death_rate=0.50 timeout_rate=0.20 stuck_rate=0.12
mean_hp_lost=14.56 mean_steps=313.62
mean_reward=+0.78 mean_time_to_kill=199.40
```

50-seed confirmation: 30% win, 50% death. The policy can reliably fight a front
AI zombie, but defense is still weak. This is the warm-start source for the next
100k easy-AI run.
