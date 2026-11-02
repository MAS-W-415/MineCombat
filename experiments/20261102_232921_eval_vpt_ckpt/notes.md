# Phase 1 probe evaluation — attack bonus 0.05

Checkpoint: `experiments/probe_attack005/checkpoints/2026-11-02_22-56-12/20/model.ckpt`  
Config: `configs/env/zombie_stage1_stationary.yaml`  
Seeds: 0–19, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.00 timeout_rate=1.00
mean_hp_lost=0.45 mean_steps=600.0 damage_ratio=5.589
episodes_with_damage=9/20 max_damage=11.8 kills=0
```

The stronger attack bonus increased engagement (9/20 episodes with damage) but
did not produce a kill. The action diagnostic showed a 49.7% attack rate, so the
policy attacked often but still missed or failed to finish.
