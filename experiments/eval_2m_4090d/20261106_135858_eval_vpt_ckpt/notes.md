# 2M checkpoint evaluation — combined 2M r1

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Protocol: 50 seeds, rep 1, max 600 steps, deterministic

```text
win_rate=0.58 death_rate=0.38 timeout_rate=0.04
mean_hp_lost=12.42 mean_steps=242.42
mean_time_to_kill=211.48275862068965 mean_reward=4.42
```

Part of the 2M flat PPO evaluation. See `docs/m3-2m-continuation.md`.
