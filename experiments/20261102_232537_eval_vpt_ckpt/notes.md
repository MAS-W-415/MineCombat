# Phase 1 probe evaluation — control

Checkpoint: `experiments/probe_control/checkpoints/2026-11-02_22-52-08/20/model.ckpt`  
Config: `configs/env/zombie_stage1_stationary.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.00 timeout_rate=1.00
mean_hp_lost=0.10 mean_steps=600.0 damage_ratio=10.0
episodes_with_damage=2/20 max_damage=18.0 kills=0
```

This is the 10k warm-start control. It dealt damage in 2/20 episodes but did
not kill.
