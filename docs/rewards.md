# Reward Design

## Formula

```text
r_t = w_hit * Δdamage_dealt
    - w_dmg * Δhp_lost
    - w_time
    + w_kill * kill
    - w_death * death
```

Stage-1 values:

| Term | Weight | Rationale |
|---|---:|---|
| `Δdamage_dealt` | 0.01 | 0.1 per HP; the stat is in tenths of a HP |
| `Δhp_lost` | 0.2 | Stronger penalty for taking damage than for dealing it |
| time | 0.001 | Small pressure against indefinite camping |
| `kill` | 10.0 | Terminal objective |
| `death` | -5.0 | Avoid suicidal policies |

The reward is clipped to `[-10, 20]` to limit the effect of event spikes.

## Event mapping

MineStudio exposes cumulative counters in `info`:

- `info["damage_dealt"]` — total damage dealt, in tenths of a HP. The engine
  stores this under `info["custom"]["damage_dealt"]`; `minecombat.envs.patches`
  makes `ObserveFromFullStats` read it. Some builds return
  `{"zombie": value}` instead of a scalar, so `event_count` handles both.
- `info["kill_entity"]["zombie"]` — total zombies killed.
- `info["health"]` — current player health.

The reward uses deltas between consecutive `info` dictionaries. `kill` and
`death` are terminal events and are counted once per episode. Because the
damage stat is cumulative across episodes, the calculator snapshots its value
at reset and only rewards the delta.

## Why not only a kill reward?

A pure kill bonus is sparse. With a randomly placed zombie and a random player
yaw, the agent may spend hundreds of steps before the first reward, which makes
PPO sample-inefficient. The dense damage term gives credit for partial progress.

The risk is reward hacking: the agent may learn to hit once and then avoid the
enemy, or spin in place. The time penalty, the HP-loss penalty, and the stuck
metric are the first defenses. Video review is required before accepting a
checkpoint.

## Why not distance shaping?

Distance-to-enemy shaping is tempting but introduces a privileged signal and a
new failure mode: the agent can farm distance reward without attacking. It is
deferred until the dense reward baseline is measured. If learning stalls, add a
potential-based distance term and ablate it.

## Ablation plan

1. Sparse kill-only reward.
2. Dense reward without HP penalty.
3. Dense reward without time penalty.
4. Dense reward with distance shaping.
5. Reward clipping on vs off.

Each ablation uses the same environment seeds and evaluation protocol.

## Debugging checklist

- Is `damage_dealt` increasing when the agent attacks?
- Is `kill_entity` incremented exactly once per kill?
- Does `health` reach zero on death?
- Are rewards roughly zero when the agent does nothing?
- Is the clip hiding large negative events?
- Does the agent receive a reward for hitting a dead or despawned zombie?

## Implementation

`src/minecombat/envs/callbacks/combat_reward.py` contains a pure
`CombatRewardCalculator` and a thin `CombatRewardCallback`. The calculator is
unit-tested in `tests/test_reward_math.py`.
