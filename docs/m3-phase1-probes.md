# M3 — Phase 1 Reward/Exploration Probes

Date: 2026-11-02  
Environment: stationary NoAI zombie, 2.5–3.5 blocks in front  
Base checkpoint: `experiments/flat_100k_stationary_attack/checkpoints/2026-11-02_19-30-36/2/`  
Probe length: 10,240 steps (warm-started), 1 environment

## Why probes

The previous 100k runs showed the policy almost never pressed attack. Rather
than running more 100k experiments, we screened six single-variable probes at
10k steps and compared action-level behavior.

## Probes

| Probe | Change |
|---|---|
| control | current stationary + attack 0.02 |
| attack005 | attack bonus 0.05 |
| attack010 | attack bonus 0.10 |
| entropy001 | entropy bonus 0.01 |
| entropy003 | entropy bonus 0.03 |
| mask | inventory masked (attack 0.02) |

## Action diagnostics (2 episodes)

| Probe | Attack rate | Inventory rate | Damage (2 eps) |
|---|---:|---:|---:|
| control | 0.0% | 0.0% | 0.0 |
| attack005 | 49.7% | 0.0% | 2.4 HP |
| attack010 | 0.0% | 0.0% | 0.0 |
| entropy001 | 0.2% | 0.0% | 0.0 |
| entropy003 | 0.0% | 0.0% | 0.0 |
| mask | 14.0% | 0.0% | 14.1 HP |

Attack bonus increased button pressing, but attack005 still did not kill.
Entropy had no meaningful effect. Inventory masking produced both a reasonable
attack rate and the highest damage.

## 20-seed evaluation (stationary task)

| Probe | Win rate | Death | Timeout | HP lost | Damage episodes | Kills |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.00 | 0.00 | 1.00 | 0.10 | 2/20 | 0 |
| attack005 | 0.00 | 0.00 | 1.00 | 0.45 | 9/20 | 0 |
| **mask** | **0.10** | 0.00 | 0.90 | 0.00 | 8/20 | **2** |

## 50-seed confirmation (mask)

```text
win_rate=0.08 death_rate=0.00 timeout_rate=0.92 stuck_rate=0.50
mean_hp_lost=0.20 mean_steps=570.42
mean_reward=+0.748 mean_time_to_kill=230.25
```

The mask probe is the first configuration with a positive mean reward and
repeatable kills: 4/50 wins, zero deaths, time-to-kill ~230 steps.

This is still the stationary NoAI task, not the final environment. The next step
is a 100k warm-start from the mask checkpoint, then a transfer to the
counterattacking zombie.

## Conclusion

- Inventory masking is the winner.
- Attack bonus alone creates attack spam without finishing.
- Entropy alone does not solve action discovery.
- The next experiment should be a longer warm-start from `probe_mask`, not a new
  from-scratch run.

## Artifacts

- `experiments/probe_control/`
- `experiments/probe_attack005/`
- `experiments/probe_attack010/`
- `experiments/probe_entropy001/`
- `experiments/probe_entropy003/`
- `experiments/probe_mask/`
- `experiments/20261102_232537_eval_vpt_ckpt/` (control, 20 seeds)
- `experiments/20261102_232921_eval_vpt_ckpt/` (attack005, 20 seeds)
- `experiments/20261102_233301_eval_vpt_ckpt/` (mask, 20 seeds)
- `experiments/20261102_234119_eval_vpt_ckpt/` (mask, 50 seeds)
