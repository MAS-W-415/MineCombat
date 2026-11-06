# 2M checkpoint evaluation — random 2M r3

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1.yaml`  
Protocol: 50 seeds, rep 3, max 600 steps, deterministic

```text
win_rate=0.14 death_rate=0.54 timeout_rate=0.32
mean_hp_lost=15.08 mean_steps=428.28
mean_time_to_kill=270 mean_reward=-1.64
```

Part of the 2M flat PPO evaluation. See `docs/m3-2m-continuation.md`.
