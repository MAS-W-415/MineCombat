# 1M checkpoint evaluation — combined r3

Checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Protocol: 50 seeds, rep 3, max 600 steps, deterministic

```text
win_rate=0.52 death_rate=0.46 timeout_rate=0.02
mean_hp_lost=14.24 mean_hp_lost_on_win=11.076923076923077
mean_steps=250.40 mean_time_to_kill=230.8846153846154
mean_reward=3.33
```

Part of the 1M flat PPO evaluation. See `docs/m3-1m-flat-ppo.md`.
