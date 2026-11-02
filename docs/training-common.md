# Common Training Preflight

This is the single source of truth for setup steps shared by flat PPO, HH1, and
HH2 runs. Experiment-specific runbooks should link here instead of repeating
these commands.

See `docs/environment.md` for the full installation and environment notes.

## Hardware baseline

| Setup | Recommended |
|---|---|
| Local smoke | 1 environment, 8 GB VRAM, 31 GB RAM |
| Full 8-environment training | 24 GB GPU (4090/cloud), 64 GB+ RAM |

Measured locally: 1 environment works at ~45 steps/s; 2 environments work but
are slower; 4 clients exhaust 31 GB system memory, not VRAM.

## Software setup

- Python 3.10
- JDK 8 at `$HOME/.local/opt/jdk8/usr/lib/jvm/java-8-openjdk-amd64`
- MineStudio installed in `.venv`
- Hugging Face mirror: `HF_ENDPOINT=https://hf-mirror.com`

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export JAVA_HOME="$HOME/.local/opt/jdk8/usr/lib/jvm/java-8-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
```

## Engine

The Minecraft engine lives in `/tmp/MineStudio` and may be cleared by the
system. Restore it before a long run:

```bash
python - <<'PY'
from minestudio.simulator.entry import check_engine
check_engine(skip_confirmation=True)
PY

python scripts/patch_minestudio.py
python scripts/patch_minestudio_engine.py
```

- `patch_minestudio.py` fixes the Malmo launch race.
- `patch_minestudio_engine.py` fixes the `leftClickCounter=10000` left-click lock.
- Both patches are idempotent.

## Ray

Ray must run with JDK 8 in its environment and the custom MineStudio resources:

- `database: 10000`
- `rollout_workers: 100`
- `wandb: 1`

```bash
bash scripts/start_ray.sh
ray status
bash scripts/stop_ray.sh
```

`start_ray.sh` sets `RAY_AUTH_MODE=disabled`, the custom resources, and the local
JDK 8. Do not start Ray from a shell with Java 11; Minecraft will fail with
`javax/xml/bind/JAXBException`.

## Preflight checks

```bash
pytest -q
ruff check .
```

For a specific trainer, also run its dry-run and a small smoke test. See the
experiment runbook for the exact command.

### Simulator smoke

```bash
python scripts/smoke_test.py \
  --config configs/env/zombie_stage1.yaml \
  --steps 100
```

Check that Minecraft starts, reset works, RGB observations arrive, rewards are
finite, and termination/truncation flags are correct.

### Damage signal validation

```bash
python scripts/test_melee.py
```

Confirm that native attacks increase `damage_dealt`, a kill is recorded, the
dense damage term is nonzero, and cumulative damage is differenced correctly
across episodes.

Common local smoke pattern:

```bash
bash scripts/train_zombie.sh --smoke --save-path experiments/smoke
```

For W&B without a login, pass:

```bash
--wandb-mode offline
```

## Checkpoints and artifacts

- Checkpoints are written under `experiments/<run>/checkpoints/<timestamp>/<update>/`.
- A flat PPO checkpoint contains `model.ckpt`, `optimizer.ckpt`, and
  `whole_config.pkl`. The model alone is ~284 MB; the optimizer is ~568 MB.
- Evaluation artifacts live under `experiments/<timestamp>_eval_*` and include
  `config.yaml`, `policy.json`, `metrics.json`, `episodes.csv`, and
  `git_commit.txt`.
- Large checkpoint files are ignored by git. Commit metrics and notes, not the
  checkpoints.

Warm-start a flat PPO run with:

```bash
--init-checkpoint experiments/<run>/checkpoints/<timestamp>/<update>/model.ckpt
```

This restores model weights only; optimizer state and iteration count start
fresh. Full resume is not implemented.

## Evaluation protocol

Use the same 50 seeds for every variant:

```bash
# Flat policy
python scripts/eval.py --policy vpt \
  --checkpoint experiments/<run>/checkpoints/<timestamp>/<update>/model.ckpt \
  --episodes 50 --seed-start 0 --max-steps 600 --deterministic

# Random baseline
python scripts/eval.py --policy random \
  --episodes 50 --seed-start 0 --max-steps 600 --deterministic

# HH1 fixed tactics
python scripts/eval.py --policy conditioned \
  --checkpoint experiments/<hh1>/checkpoints/<timestamp>/<update>/model.ckpt \
  --num-tactics 2 --tactic-ids 0 1 \
  --episodes 50 --seed-start 0 --max-steps 600 --deterministic
```

`scripts/eval_baseline.py` and `scripts/eval_conditioned.py` remain as
compatibility wrappers.

Report win rate, death rate, timeout/stuck rate, mean HP lost, mean steps, and
damage ratio. Damage `damage_dealt` is normalized to HP in the evaluation
runner; Minecraft stores it in tenths of a HP internally.

### Variance warning

Minecraft's mob AI is not reset between episodes by the fast reset path. The
same checkpoint, config, and seeds can differ by 10+ percentage points across
repeated evaluations (observed range 10–24% on the combined task). Spawn
distance and angle are deterministic and are now recorded in `episodes.csv`.

For any claim or gate measurement:

- run three independent 50-seed evaluations and report the mean and range;
- use 100+ seeds for the final gate;
- label 20-seed runs as screens, never as results.

See `docs/m3-evaluation-variance.md` for the reproduction.

## Cleanup

After every run, crashed or successful:

```bash
bash scripts/stop_ray.sh
pgrep -f '[m]cprec-6.13.jar' | xargs -r kill -9
nvidia-smi
```

Orphaned Java clients keep rendering at 100% CPU and hold GPU memory.

## Common failures

| Symptom | Cause / fix |
|---|---|
| `javax/xml/bind/JAXBException` | Ray was started with Java 11; restart with JDK 8 |
| Engine prompt / `EOFError` in EnvWorker | Engine missing; run `check_engine(skip_confirmation=True)` |
| `No API key configured` (wandb) | Use `--wandb-mode offline` or `wandb login` |
| 4 environments crash | 31 GB system RAM exhausted; use 2 or fewer locally |
| Two environments slower | Each client competes for CPU/GPU; keep 1 for local smoke |
| `too high kl` | PPO trust-region warning; check losses and do not overrun |
