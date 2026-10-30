# M1 Melee Blocker — `attack` Does Not Register

Date: 2026-10-30  
Environment: MineStudio 1.1.6, engine `mcprec-6.13.jar`, RTX 4060

## Summary

The native `attack` action (left mouse button) does not damage entities in the
current MineStudio engine. The native `use` action (right mouse button) works:
a bow shoots and arrows are consumed.

This blocks the planned zombie melee stage. It is not a reward or raycast issue
in MineCombat: the action reaches the engine as `attack 1`, but no melee damage
is applied.

## Reproduction

```bash
python scripts/test_melee.py --steps 300
```

The script:

1. builds a flat platform,
2. faces the player at yaw 0,
3. summons a stationary cow/zombie in front,
4. alternates `attack` press/release,
5. checks `damage_dealt` and `kill_entity`.

Result:

```text
RESULT damage=0.0 kills=0.0 outcome=ongoing passed=False
```

## What works

- `use` with a bow: `use_item` increments and arrow count drops.
- Movement: `forward` changes `player_pos`.
- Camera: `camera` changes yaw/pitch.
- Commands: summon, fill, teleport, give all work.

## What does not work

- `attack` against a cow or zombie: `damage_dealt` stays empty.
- `attack` against a placed dirt block: `mine_block`/`break_item` stay empty.
- Holding attack, alternating press/release, and 300-step runs all fail.
- Writing `key_key.attack:key.mouse.left` into the runtime `options.txt` does
  not change the result.

## Action pipeline evidence

The agent action translates correctly:

```text
translated attack: array([1])
command string: ... attack 1 ... use 0 ...
```

The engine maps:

- `attack` → mouse button 0 (`actionToMouseButton`)
- `use` → mouse button 1

So the command is reaching the engine, but the left-click path is not producing
a melee hit.

## Suspected area

The engine's `MouseHelper` / `ReplaySender` path:

- `constructMouseState` adds both attack and use to the same button set.
- `ReplaySender.execMouseButtons` calls `mouseButtonCallbackImpl`.
- `Minecraft.runTick` only calls `clickMouse()` when
  `keyBindAttack.isPressed()` is true.
- `clickMouse()` is gated by `leftClickCounter` and `objectMouseOver`.

Possible causes:

1. `leftClickCounter` stays above zero.
2. `objectMouseOver` is null or not the entity.
3. The left-click key-press transition is consumed before the client tick.
4. The engine's mouse-button state is not applied to the live client.

## Options

1. **Fix the engine** (preferred for the research goal): decompile/patch the
   engine or use a Minecraft version/engine where left-click works.
2. **Use a bow for Stage 1** (native `use`, clean experiment, but changes the
   melee research question).
3. **Temporary command-damage workaround** (keeps the `attack` action but applies
   damage via a server command; not native melee, should be clearly labelled).

Do not silently replace native melee with command damage in a research run.
