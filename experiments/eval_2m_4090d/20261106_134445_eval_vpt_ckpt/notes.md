# 2M checkpoint evaluation — angle30 2M

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Protocol: 1x50, max 600 steps, deterministic

```text
win_rate=0.64 death_rate=0.30 timeout_rate=0.06
mean_hp_lost=11.34 mean_steps=222.74
mean_time_to_kill=188.96875 mean_reward=5.41
```

Part of the 2M flat PPO evaluation. See `docs/m3-2m-continuation.md`.
