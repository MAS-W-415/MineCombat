# 1M checkpoint evaluation — random r1

Checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Config: `configs/env/zombie_stage1.yaml`  
Protocol: 50 seeds, rep 1, max 600 steps, deterministic

```text
win_rate=0.22 death_rate=0.62 timeout_rate=0.16
mean_hp_lost=15.48 mean_hp_lost_on_win=12.272727272727273
mean_steps=375.62 mean_time_to_kill=291.45454545454544
mean_reward=-0.62
```

Part of the 1M flat PPO evaluation. See `docs/m3-1m-flat-ppo.md`.
