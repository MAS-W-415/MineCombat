# M3 — 100k Stationary-Target Curriculum

Date: 2026-11-02  
Config: `configs/env/zombie_stage1_stationary.yaml`  
Checkpoint: `experiments/flat_100k_stationary/checkpoints/2026-11-02_18-38-41/2/model.ckpt`

## Motivation

The easy curriculum put the zombie in front, but the policy still rarely
attacked and was often killed at close range. The stationary-target stage
removes the counterattack:

- NoAI zombie
- 2.5–3.5 blocks directly in front
- player yaw 0
- baseline reward

If the policy still cannot kill this target, the bottleneck is the attack
action itself, not enemy search or survival pressure.

## Results

### On the stationary task (50 seeds)

| Metric | Value |
|---|---:|
| Win rate | 0.00 |
| Death rate | 0.00 |
| Timeout rate | 1.00 |
| Stuck rate | 0.96 |
| Mean HP lost | 0.08 |
| Mean steps | 600.0 |
| Mean damage | ~0.24 HP/episode |

The reported `damage_ratio=2.95` is misleading because HP loss is almost zero.
The policy dealt very little damage and never killed the stationary zombie.

### On the real random-spawn task (50 seeds)

| Metric | Value |
|---|---:|
| Win rate | 0.00 |
| Death rate | 0.32 |
| Timeout rate | 0.68 |
| Mean HP lost | 10.86 |
| Damage ratio | 0.0459 |

Transfer to the real environment also failed.

## Action-level diagnosis

`scripts/diagnose_attack.py` counts the selected button combinations:

```text
1,200 steps: 10 attack presses (0.8%)
1 episode (600 steps): 0 attack presses
```

The policy mostly rotates the camera and strafes. It almost never presses the
attack button, even when a stationary zombie is directly in front.

## Conclusion

The bottleneck is **action discovery**, not target visibility, reward scale, or
enemy difficulty. Reward-only and curriculum-only changes cannot fix a policy
that never selects the attack action.

## Next options

1. **Attack-attempt shaping**: add a small bonus for pressing attack, with a
   cooldown to avoid spamming. This directly increases attack exploration.
2. **Demonstration / behavior cloning**: record scripted attacks and warm-start
   the policy to imitate them.
3. **Action-space simplification**: restrict stage 1 to a smaller combat action
   set (camera, forward/back/strafe, attack) instead of the full 8,641-combo
   space.

Attack-attempt shaping is the cheapest next experiment.

## Artifacts

- `experiments/20261102_191013_eval_vpt_ckpt/` (random spawn)
- `experiments/20261102_191835_eval_vpt_ckpt/` (stationary)
- `scripts/diagnose_attack.py`
