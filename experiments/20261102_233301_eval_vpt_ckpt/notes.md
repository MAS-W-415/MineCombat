# Phase 1 probe evaluation — inventory mask (20 seeds)

Checkpoint: `experiments/probe_mask/checkpoints/2026-11-02_23-10-56/20/model.ckpt`  
Config: `configs/env/zombie_stage1_stationary.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.10 death_rate=0.00 timeout_rate=0.90
mean_hp_lost=0.00 mean_steps=562.2
episodes_with_damage=8/20 max_damage=20.1 kills=2
```

The inventory mask is the first probe with kills: 2/20 wins, zero damage taken,
and 8/20 episodes with damage. This was promoted to a 50-seed confirmation.
