# 1M checkpoint evaluation — combined r1

Checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Protocol: 50 seeds, rep 1, max 600 steps, deterministic

```text
win_rate=0.32 death_rate=0.66 timeout_rate=0.02
mean_hp_lost=15.00 mean_hp_lost_on_win=9.375
mean_steps=250.20 mean_time_to_kill=228.1875
mean_reward=0.99
```

Part of the 1M flat PPO evaluation. See `docs/m3-1m-flat-ppo.md`.
