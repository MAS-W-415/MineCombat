# 100k easy-AI mask warm-start — easy AI (50 seeds)

Checkpoint: `experiments/flat_100k_easy_mask_warm/checkpoints/2026-11-03_00-33-34/2/`  
Config: `configs/env/zombie_stage1_easy.yaml` (front spawn, AI zombie)  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.30 death_rate=0.58 timeout_rate=0.12 stuck_rate=0.08
mean_hp_lost=14.36 mean_steps=287.34
mean_reward=+1.10 mean_time_to_kill=187.87
```

Win rate is unchanged from the previous easy checkpoint (30%), but deaths rose
from 50% to 58% and kills are faster (188 vs 199 steps). The policy became more
aggressive at the cost of survival.
