# 2M checkpoint evaluation — easy 2M

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1_easy.yaml`  
Protocol: 1x50, max 600 steps, deterministic

```text
win_rate=0.90 death_rate=0.10 timeout_rate=0.00
mean_hp_lost=6.96 mean_steps=178.86
mean_time_to_kill=166.66666666666666 mean_reward=9.30
```

Part of the 2M flat PPO evaluation. See `docs/m3-2m-continuation.md`.
