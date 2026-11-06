# 2M checkpoint evaluation — combined 2M r3

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Protocol: 50 seeds, rep 3, max 600 steps, deterministic

```text
win_rate=0.62 death_rate=0.32 timeout_rate=0.06
mean_hp_lost=11.68 mean_steps=257.72
mean_time_to_kill=204.32258064516128 mean_reward=5.15
```

Part of the 2M flat PPO evaluation. See `docs/m3-2m-continuation.md`.
