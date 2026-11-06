# 2M checkpoint evaluation — random 2M r1

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1.yaml`  
Protocol: 50 seeds, rep 1, max 600 steps, deterministic

```text
win_rate=0.06 death_rate=0.54 timeout_rate=0.40
mean_hp_lost=14.72 mean_steps=461.32
mean_time_to_kill=288.3333333333333 mean_reward=-2.60
```

Part of the 2M flat PPO evaluation. See `docs/m3-2m-continuation.md`.
