# M3 — 100k Stationary Target with Attack Shaping

Date: 2026-11-02  
Config: `configs/env/zombie_stage1_stationary_attack.yaml`  
Checkpoint: `experiments/flat_100k_stationary_attack/checkpoints/2026-11-02_19-30-36/2/model.ckpt`

## Motivation

The stationary-target baseline pressed attack in <1% of steps and never killed
the NoAI zombie. This run adds a small attack-attempt bonus:

```text
attack: 0.02
attack_cooldown_steps: 10
```

The bonus is action-discovery shaping; it is not a claim that the policy learned
combat.

## Result

50 seeds on the stationary task:

| Metric | Stationary baseline | Stationary + attack shaping |
|---|---:|---:|
| Win rate | 0.00 | 0.00 |
| Timeout rate | 1.00 | 1.00 |
| Stuck rate | 0.96 | 0.86 |
| Mean HP lost | 0.08 | 0.56 |
| Episodes with damage | 0/50 | **7/50** |
| Mean damage | ~0.24 HP | **0.94 HP** |
| Max damage | ~0 | **11.8 HP** |

The policy now attacks sometimes (7/50 episodes) and can take a stationary
zombie down to ~8 HP, but it never finishes the kill. All 50 episodes timed out.

## Interpretation

Attack-attempt shaping moved the policy in the right direction, but 100k steps
is not enough to turn occasional hits into a reliable kill sequence. The policy
still spends most steps rotating the camera and strafing.

## Next options

1. **300k stationary + attack shaping**: same config, longer run (~75 minutes).
   Tests whether the 7/50 damage rate becomes kills.
2. **Stronger attack bonus** (`0.05`) or an entropy bonus: increases exploration
   of the attack action, but risks attack spam.
3. **Demonstration / behavior cloning**: show the policy a scripted attack
   sequence, then fine-tune with PPO.

Option 1 is the cleanest next step: it changes one variable (training length)
and uses the current positive signal.

## Artifacts

- `experiments/20261102_200352_eval_vpt_ckpt/`
- `scripts/diagnose_attack.py`
