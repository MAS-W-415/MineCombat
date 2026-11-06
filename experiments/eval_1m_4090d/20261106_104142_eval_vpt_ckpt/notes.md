# 1M checkpoint evaluation — combined r2

Checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Protocol: 50 seeds, rep 2, max 600 steps, deterministic

```text
win_rate=0.38 death_rate=0.58 timeout_rate=0.04
mean_hp_lost=14.46 mean_hp_lost_on_win=10.263157894736842
mean_steps=276.24 mean_time_to_kill=223.31578947368422
mean_reward=1.68
```

Part of the 1M flat PPO evaluation. See `docs/m3-1m-flat-ppo.md`.
