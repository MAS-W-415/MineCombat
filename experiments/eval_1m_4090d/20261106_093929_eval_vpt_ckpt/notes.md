# 1M checkpoint evaluation — easy

Checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Config: `configs/env/zombie_stage1_easy.yaml`  
Protocol: 1x50, max 600 steps, deterministic

```text
win_rate=0.72 death_rate=0.28 timeout_rate=0.00
mean_hp_lost=12.00 mean_hp_lost_on_win=9.666666666666666
mean_steps=193.86 mean_time_to_kill=190.91666666666666
mean_reward=6.21
```

Part of the 1M flat PPO evaluation. See `docs/m3-1m-flat-ppo.md`.
