# 2M checkpoint evaluation — random 2M r2

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1.yaml`  
Protocol: 50 seeds, rep 2, max 600 steps, deterministic

```text
win_rate=0.10 death_rate=0.60 timeout_rate=0.30
mean_hp_lost=15.00 mean_steps=449.96
mean_time_to_kill=270 mean_reward=-2.18
```

Part of the 2M flat PPO evaluation. See `docs/m3-2m-continuation.md`.
