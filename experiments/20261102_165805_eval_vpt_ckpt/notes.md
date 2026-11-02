# 100k flat PPO — 50-seed evaluation

Checkpoint: `experiments/flat_100k/checkpoints/2026-11-02_16-10-12/2/model.ckpt`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.16 timeout_rate=0.84 stuck_rate=0.78
mean_hp_lost=10.28 mean_steps=579.18 mean_reward=-2.60
damage_ratio=0.0313
```

Only 4/50 episodes dealt damage; max 4.9 HP; no kills. The policy survives
slightly better than zero-shot but mostly times out. See
`docs/m3-100k-eval.md`.
