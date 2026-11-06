# 1M checkpoint evaluation — random r3

Checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Config: `configs/env/zombie_stage1.yaml`  
Protocol: 50 seeds, rep 3, max 600 steps, deterministic

```text
win_rate=0.20 death_rate=0.46 timeout_rate=0.34
mean_hp_lost=14.98 mean_hp_lost_on_win=13.2
mean_steps=396.52 mean_time_to_kill=266.3
mean_reward=-0.80
```

Part of the 1M flat PPO evaluation. See `docs/m3-1m-flat-ppo.md`.
