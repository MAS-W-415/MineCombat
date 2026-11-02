# 100k mask warm-start — random spawn (20 seeds)

Checkpoint: `experiments/flat_100k_mask_warm/checkpoints/2026-11-02_23-43-23/2/`  
Config: `configs/env/zombie_stage1.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.50 timeout_rate=0.50
mean_hp_lost=14.15 mean_reward=-3.26 damage_ratio=0.043
```

Random spawn still fails: the policy can fight a visible front target but does
not reliably find or aim at a random one.
