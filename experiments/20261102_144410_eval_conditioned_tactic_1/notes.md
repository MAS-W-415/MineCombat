# H1 defensive tactic — 50-seed evaluation

Checkpoint: `experiments/h1_tactical_10k/checkpoints/2026-11-02_14-08-52/20/model.ckpt`  
Tactic: 1 (defensive)  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.32 timeout_rate=0.68 stuck_rate=0.66
mean_hp_lost=11.24 mean_steps=534.62 mean_reward=-2.72
damage_ratio=0.0548
```

The defensive condition took less damage and died less often than aggressive,
but dealt less damage and still did not win at 10k steps.
