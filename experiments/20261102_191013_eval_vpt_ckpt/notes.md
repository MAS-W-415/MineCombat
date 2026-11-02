# 100k stationary-target checkpoint — default environment

Checkpoint: `experiments/flat_100k_stationary/checkpoints/2026-11-02_18-38-41/2/model.ckpt`  
Evaluation config: `configs/env/zombie_stage1.yaml` (random ring spawn, AI zombie)  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.32 timeout_rate=0.68 stuck_rate=0.66
mean_hp_lost=10.86 mean_steps=539.42 mean_reward=-2.66
damage_ratio=0.0459
```

Transfer test on the real target environment: still no wins. This checkpoint was
trained only on a stationary zombie; it does not solve the random-spawn task.
