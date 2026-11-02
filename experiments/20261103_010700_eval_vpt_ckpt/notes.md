# 100k easy-AI mask warm-start — random spawn (20 seeds)

Checkpoint: `experiments/flat_100k_easy_mask_warm/checkpoints/2026-11-03_00-33-34/2/`  
Config: `configs/env/zombie_stage1.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.65 timeout_rate=0.35
mean_hp_lost=15.45 mean_reward=-3.30 damage_ratio=0.174
```

Random spawn is still unsolved: 0 wins, 65% deaths. The policy can fight a
visible front target but cannot reliably find or handle a random one.
