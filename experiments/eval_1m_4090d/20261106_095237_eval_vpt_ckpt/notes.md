# 1M checkpoint evaluation — angle30

Checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Config: `configs/env/zombie_stage1_easy_angle30_mask.yaml`  
Protocol: 1x50, max 600 steps, deterministic

```text
win_rate=0.52 death_rate=0.46 timeout_rate=0.02
mean_hp_lost=12.54 mean_hp_lost_on_win=7.961538461538462
mean_steps=222.94 mean_time_to_kill=196.53846153846155
mean_reward=3.69
```

Part of the 1M flat PPO evaluation. See `docs/m3-1m-flat-ppo.md`.
