# 2M checkpoint evaluation — combined 2M r2

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Protocol: 50 seeds, rep 2, max 600 steps, deterministic

```text
win_rate=0.66 death_rate=0.32 timeout_rate=0.02
mean_hp_lost=12.35 mean_steps=232.54
mean_time_to_kill=220.3939393939394 mean_reward=5.37
```

Part of the 2M flat PPO evaluation. See `docs/m3-2m-continuation.md`.
