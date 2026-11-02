# Defense probe — stronger HP-loss penalty (50 seeds)

Checkpoint: `experiments/probe_easy_defense/checkpoints/2026-11-03_01-14-18/20/`  
Training config: `configs/env/zombie_stage1_easy_mask_defense.yaml`  
Eval config: `configs/env/zombie_stage1_easy.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.26 death_rate=0.68 timeout_rate=0.06 stuck_rate=0.04
mean_hp_lost=15.96 mean_hp_lost_on_win=11.08
mean_steps=256.22 mean_time_to_kill=214.15 mean_reward=+0.22
```

Raising `damage_taken` from 0.2 to 0.3 did not improve survival. The action
diagnostic showed an 89% attack-button rate, so the policy became more
aggressive rather than more defensive. Reject this direction.
