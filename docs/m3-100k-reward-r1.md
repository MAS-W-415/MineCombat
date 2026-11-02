# M3 — 100k Reward Ablation R1

Date: 2026-11-02  
Config: `configs/env/zombie_stage1_aggressive.yaml`  
Checkpoint: `experiments/flat_100k_r1/checkpoints/2026-11-02_17-11-28/2/model.ckpt`  
Seeds: 0–49, max 600 steps, deterministic

## Motivation

The baseline 100k policy mostly timed out (84%) and rarely attacked. The
baseline reward made passive behavior cheap:

```text
kill +10; damage +0.1/HP; damage taken -0.2/HP; time -0.001/step
```

R1 changed the trade-off:

| Term | Baseline | R1 |
|---|---:|---:|
| kill | 10 | 20 |
| damage dealt | 0.01/stat (0.1/HP) | 0.02/stat (0.2/HP) |
| damage taken | 0.2/HP | 0.05/HP |
| death | -5 | -10 |
| time | 0.001 | 0.002 |
| clip | [-10, 20] | [-20, 40] |

## Result

| Metric | Baseline 100k | R1 100k |
|---|---:|---:|
| Win rate | 0.00 | 0.00 |
| Death rate | 0.16 | 0.36 |
| Timeout rate | 0.84 | 0.64 |
| Mean HP lost | 10.28 | 11.68 |
| Mean steps | 579.18 | 541.08 |
| Damage ratio | 0.0313 | 0.0168 |
| Episodes with damage | 4/50 | 2/50 |
| Max damage in an episode | 4.9 HP | 4.9 HP |

## Interpretation

R1 did not improve combat. It reduced timeout and increased deaths, but damage
actually fell. The policy became more willing to move/engage in a broad sense,
but still did not find and attack the zombie reliably.

The training video shows wandering and terrain viewing, no zombie approach.
This suggests the bottleneck is not only reward scale. The agent often does not
get the zombie into view or does not discover the attack sequence.

## Conclusion

Do not spend the 1M run on this reward change. The next experiment should target
exploration/visibility, not reward magnitude:

1. **Curriculum spawn**: put the zombie closer and in front for early training,
   then restore the random ring spawn.
2. **Action mask**: remove inventory/GUI actions if they still appear.
3. **Longer run with the original reward** is lower priority until the agent
   reliably engages.

Artifacts:

- `experiments/20261102_174339_eval_vpt_ckpt/`
- Training video: `output/videos/2026-11-02_17-10-03/51200 - ...mp4`
