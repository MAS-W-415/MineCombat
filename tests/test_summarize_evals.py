"""Offline evaluation summaries need only CSV/JSON artifacts and stdlib."""

from __future__ import annotations

import builtins
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_evals.py"
REQUIRED_COLUMNS = (
    "seed", "win", "death", "timeout", "steps", "hp_lost",
    "damage_dealt", "time_to_kill", "reward",
)
OPTIONAL_COLUMNS = ("outcome", "spawn_distance", "spawn_angle")
POLICY = {
    "policy": "vpt",
    "model": "CraftJarvis/MineStudio_VPT.foundation_model_1x",
    "checkpoint": "experiments/flat_2m/checkpoints/20/model.ckpt",
}
CONFIG = "arena:\n  seed: 0\n"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclasses can resolve their module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def summarizer():
    return _load_script("minecombat_test_summarize_evals")


def _row(seed=0, **overrides):
    row = {
        "seed": seed,
        "win": "True",
        "death": "False",
        "timeout": "False",
        "steps": 10,
        "hp_lost": 1.0,
        "damage_dealt": 20.0,
        "time_to_kill": 10,
        "reward": 1.0,
        "outcome": "",
        "spawn_distance": "",
        "spawn_angle": "",
    }
    row.update(overrides)
    return row


def _write_run(
    parent,
    name,
    rows,
    *,
    metrics=None,
    metadata=True,
    policy=None,
    config=CONFIG,
    columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS,
):
    run = parent / name
    run.mkdir()
    with (run / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if metrics is not None:
        (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if metadata:
        (run / "policy.json").write_text(
            json.dumps(POLICY if policy is None else policy), encoding="utf-8",
        )
        (run / "config.yaml").write_text(config, encoding="utf-8")
        (run / "git_commit.txt").write_text("578ad888\n", encoding="utf-8")
    return run


def test_repeated_seed_sets_remain_independent_evaluations(summarizer, tmp_path):
    first = _write_run(tmp_path, "first", [
        _row(0, hp_lost=2, reward=5),
        _row(1, win="False", death="True", time_to_kill="",
             hp_lost=20, damage_dealt=0, reward=-3),
        _row(2, win="False", death="True", time_to_kill="",
             hp_lost=20, damage_dealt=5, reward=-4),
        _row(3, win="False", timeout="True", time_to_kill="",
             hp_lost=5, damage_dealt=0, reward=-2),
    ])
    second = _write_run(tmp_path, "second", [
        _row(0, hp_lost=1, reward=2),
        _row(1, steps=30, time_to_kill=30, hp_lost=3, reward=4),
        _row(2, win="False", timeout="True", time_to_kill="",
             hp_lost=0, damage_dealt=0, reward=-1),
        _row(3, win="False", timeout="True", time_to_kill="",
             hp_lost=4, damage_dealt=2, reward=-1),
    ])

    report = summarizer.summarize_runs([first, second], task="combined")

    assert report["schema_version"] == 1
    assert report["task"] == "combined"
    assert [run["run_id"] for run in report["runs"]] == ["first", "second"]
    assert [run["episodes"] for run in report["runs"]] == [4, 4]
    assert Path(report["runs"][0]["path"]) == first.resolve()
    assert report["aggregate"]["runs"] == 2
    assert report["aggregate"]["episodes"] == 8
    assert report["aggregate"]["unique_seeds"] == 4
    per_run = report["aggregate"]["per_run"]
    win_stats = {key: per_run["win_rate"][key] for key in ("mean", "min", "max")}
    assert win_stats == pytest.approx({
        "mean": 0.375, "min": 0.25, "max": 0.5,
    })
    assert per_run["mean_hp_lost"]["mean"] == pytest.approx(6.875)
    assert per_run["mean_reward"]["mean"] == pytest.approx(0.0)
    assert per_run["mean_time_to_kill"]["mean"] == pytest.approx(15.0)
    # These fractions use every episode, not only deaths or timeouts.
    assert report["runs"][0]["metrics"]["zero_damage_death_rate"] == 0.25
    assert report["runs"][0]["metrics"]["zero_damage_timeout_rate"] == 0.25
    assert per_run["zero_damage_death_rate"]["mean"] == 0.125
    assert per_run["zero_damage_timeout_rate"]["mean"] == 0.25


@pytest.mark.parametrize("truth,falsehood", [("TRUE", "false"), ("true", "FALSE"), ("1", "0")])
def test_boolean_spellings_and_optional_columns(summarizer, tmp_path, truth, falsehood):
    run = _write_run(
        tmp_path, "run", [_row(win=truth, death=falsehood, timeout=falsehood)],
        columns=REQUIRED_COLUMNS,
    )
    report = summarizer.summarize_runs([run], task="easy")
    assert report["runs"][0]["metrics"]["win_rate"] == 1.0


def test_no_kill_time_is_unknown_not_zero(summarizer, tmp_path):
    run = _write_run(tmp_path, "run", [
        _row(win="False", timeout="True", time_to_kill="", reward=-2),
    ])
    result = summarizer.summarize_runs([run], task="random")["runs"][0]
    assert result["metrics"]["mean_time_to_kill"] is None
    assert result["metrics"]["mean_reward"] == -2.0


@pytest.mark.parametrize("missing_time", ["", "  ", None])
def test_partial_wins_missing_kill_time_are_rejected(summarizer, tmp_path, missing_time):
    run = _write_run(tmp_path, "partial_kill_times", [
        _row(0),
        _row(1, time_to_kill=missing_time),
    ])
    with pytest.raises(summarizer.SummaryError) as error:
        summarizer.summarize_runs([run], task="easy")
    assert f"{run.resolve() / 'episodes.csv'}:3" in str(error.value)
    assert "time_to_kill" in str(error.value)
    assert "win" in str(error.value)


def test_external_completion_does_not_need_a_win_death_or_timeout(summarizer, tmp_path):
    run = _write_run(tmp_path, "run", [
        _row(win="False", time_to_kill="", outcome="external_done"),
    ])
    result = summarizer.summarize_runs([run], task="easy")["runs"][0]
    assert result["metrics"]["win_rate"] == 0.0
    assert result["metrics"]["death_rate"] == 0.0
    assert result["metrics"]["timeout_rate"] == 0.0


@pytest.mark.parametrize("flag", ["win", "death", "timeout"])
def test_external_completion_cannot_also_have_an_outcome_flag(summarizer, tmp_path, flag):
    values = {"win": "False", "time_to_kill": "", "outcome": "external_done", flag: "True"}
    if flag == "win":
        values["time_to_kill"] = 10
    run = _write_run(tmp_path, "run", [_row(**values)])
    with pytest.raises(summarizer.SummaryError, match="external_done conflicts"):
        summarizer.summarize_runs([run], task="easy")


def test_misspelled_outcome_is_rejected(summarizer, tmp_path):
    run = _write_run(tmp_path, "run", [_row(outcome="sucess")])
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


def test_empty_artifact_metadata_stays_unknown(summarizer, tmp_path):
    unknown = _write_run(tmp_path, "unknown", [_row()], metadata=False)
    known = _write_run(tmp_path, "known", [_row()])
    report = summarizer.summarize_runs([unknown, known], task="easy")
    result = report["runs"][0]
    expected = {
        "policy": None, "model": None, "checkpoint": None,
        "config_sha256": None, "git_commit": None,
    }
    assert {key: result["metadata"][key] for key in expected} == expected
    assert result["warnings"]
    assert report["aggregate"]["episodes"] == 2
    assert report["runs"][1]["metadata"]["config_sha256"] == hashlib.sha256(
        CONFIG.encode("utf-8"),
    ).hexdigest()
    assert report["runs"][1]["metadata"]["git_commit"] == "578ad888"


@pytest.mark.parametrize("filename", ["metrics.json", "policy.json", "config.yaml", "git_commit.txt"])
def test_missing_optional_artifacts_warn(summarizer, tmp_path, filename):
    run = _write_run(tmp_path, "run", [_row()], metrics={"episodes": 1})
    (run / filename).unlink()
    result = summarizer.summarize_runs([run], task="easy")["runs"][0]
    assert result["warnings"]
    if filename == "metrics.json":
        assert any("metrics.json" in warning for warning in result["warnings"])
    else:
        key = {
            "policy.json": "policy", "config.yaml": "config_sha256",
            "git_commit.txt": "git_commit",
        }[filename]
        assert result["metadata"][key] is None


@pytest.mark.parametrize("field", ["policy", "model", "checkpoint"])
def test_conflicting_known_policy_metadata_is_rejected(summarizer, tmp_path, field):
    first = _write_run(tmp_path, "first", [_row()])
    changed = {**POLICY, field: "different"}
    second = _write_run(tmp_path, "second", [_row()], policy=changed)
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([first, second], task="easy")


def test_explicit_no_checkpoint_cannot_mix_with_checkpoint(summarizer, tmp_path):
    pretrained = _write_run(tmp_path, "pretrained", [_row()], policy={
        **POLICY, "checkpoint": None,
    })
    trained = _write_run(tmp_path, "trained", [_row()])
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([pretrained, trained], task="easy")


def test_absent_checkpoint_field_is_unknown_and_can_mix_with_known(summarizer, tmp_path):
    unknown_policy = {key: value for key, value in POLICY.items() if key != "checkpoint"}
    unknown = _write_run(tmp_path, "unknown", [_row()], policy=unknown_policy)
    trained = _write_run(tmp_path, "trained", [_row()])
    report = summarizer.summarize_runs([unknown, trained], task="easy")
    assert report["runs"][0]["metadata"]["checkpoint"] is None
    assert report["runs"][0]["warnings"]
    assert report["aggregate"]["runs"] == 2


@pytest.mark.parametrize("field,value", [
    ("deterministic", "false"),
    ("deterministic", 0),
    ("num_tactics", 0),
    ("num_tactics", -1),
    ("num_tactics", 1.5),
    ("num_tactics", True),
    ("max_steps", 0),
    ("max_steps", -1),
    ("max_steps", 1.0),
    ("max_steps", True),
    ("tactic_id", -1),
    ("tactic_id", 1.5),
    ("tactic_id", True),
])
def test_invalid_policy_metadata_types_are_rejected(summarizer, tmp_path, field, value):
    run = _write_run(tmp_path, "run", [_row()], policy={**POLICY, field: value})
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


def test_valid_policy_metadata_types_are_accepted(summarizer, tmp_path):
    run = _write_run(tmp_path, "run", [_row()], policy={
        **POLICY, "deterministic": False, "num_tactics": 2,
        "max_steps": 600, "tactic_id": 0,
    })
    report = summarizer.summarize_runs([run], task="easy")
    assert report["aggregate"]["episodes"] == 1


def test_episode_steps_equal_to_known_limit_are_accepted(summarizer, tmp_path):
    run = _write_run(tmp_path, "at_limit", [
        _row(0),
        _row(1, steps=600, time_to_kill=600),
    ], policy={**POLICY, "max_steps": 600})
    report = summarizer.summarize_runs([run], task="easy")
    assert report["aggregate"]["episodes"] == 2
    assert report["runs"][0]["metrics"]["mean_steps"] == 305


def test_later_episode_exceeding_known_limit_is_rejected(summarizer, tmp_path):
    run = _write_run(tmp_path, "over_limit", [
        _row(0),
        _row(7, steps=601, time_to_kill=601),
    ], policy={**POLICY, "max_steps": 600})
    with pytest.raises(summarizer.SummaryError) as error:
        summarizer.summarize_runs([run], task="easy")
    message = str(error.value)
    assert str(run.resolve()) in message
    assert "seed=7" in message
    assert "steps=601" in message
    assert "max_steps=600" in message


@pytest.mark.parametrize("limit_metadata", [{}, {"max_steps": None}], ids=["missing", "null"])
@pytest.mark.parametrize("known_first", [False, True])
def test_unknown_step_limit_is_not_inferred(summarizer, tmp_path, limit_metadata, known_first):
    config = CONFIG + "max_steps: 600\n"
    unknown = _write_run(tmp_path, "unknown_limit", [
        _row(0, steps=601, time_to_kill=601),
    ], policy={**POLICY, **limit_metadata}, config=config)
    known = _write_run(tmp_path, "known_limit", [_row(0)],
                       policy={**POLICY, "max_steps": 600}, config=config)
    paths = [known, unknown] if known_first else [unknown, known]
    report = summarizer.summarize_runs(paths, task="easy")
    result = next(run for run in report["runs"] if run["run_id"] == "unknown_limit")
    assert result["metadata"]["max_steps"] is None
    assert "max_steps" not in result["known_metadata_fields"]
    assert result["metrics"]["mean_steps"] == 601
    assert any("max_steps" in warning for warning in result["warnings"])
    assert report["aggregate"]["episodes"] == 2


@pytest.mark.parametrize("field", ["config_sha256", "git_commit"])
def test_policy_metadata_cannot_conflict_with_adjacent_artifact(summarizer, tmp_path, field):
    run = _write_run(tmp_path, "run", [_row()], policy={
        **POLICY, field: "inconsistent-recorded-identity",
    })
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


@pytest.mark.parametrize("field,value", [
    ("config_sha256", hashlib.sha256(CONFIG.encode("utf-8")).hexdigest()),
    ("git_commit", "578ad888"),
])
def test_matching_policy_and_adjacent_metadata_are_accepted(summarizer, tmp_path, field, value):
    run = _write_run(tmp_path, "run", [_row()], policy={**POLICY, field: value})
    report = summarizer.summarize_runs([run], task="easy")
    assert report["runs"][0]["metadata"][field] == value


@pytest.mark.parametrize("field,known_value", [
    ("policy", POLICY["policy"]),
    ("model", POLICY["model"]),
    ("deterministic", False),
    ("num_tactics", 2),
    ("max_steps", 600),
    ("tactic_id", 0),
])
def test_null_noncheckpoint_metadata_is_unknown(summarizer, tmp_path, field, known_value):
    unknown = _write_run(tmp_path, "unknown", [_row()], policy={**POLICY, field: None})
    known = _write_run(tmp_path, "known", [_row()], policy={**POLICY, field: known_value})
    report = summarizer.summarize_runs([unknown, known], task="easy")
    assert report["runs"][0]["metadata"][field] is None
    assert report["runs"][0]["warnings"]
    assert report["aggregate"]["runs"] == 2


def test_config_comparison_uses_exact_file_content(summarizer, tmp_path):
    first = _write_run(tmp_path, "first", [_row()])
    second = _write_run(tmp_path, "second", [_row()], config=CONFIG + "\n")
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([first, second], task="easy")


def test_conflicting_known_git_commits_are_rejected(summarizer, tmp_path):
    first = _write_run(tmp_path, "first", [_row()])
    second = _write_run(tmp_path, "second", [_row()])
    (second / "git_commit.txt").write_text("another-commit\n", encoding="utf-8")
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([first, second], task="easy")


@pytest.mark.parametrize("alias", [False, True])
def test_same_run_cannot_be_counted_twice(summarizer, tmp_path, alias):
    run = _write_run(tmp_path, "run", [_row()])
    repeated = run / ".." / run.name if alias else run
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run, repeated], task="easy")


def test_duplicate_seed_inside_one_run_is_rejected(summarizer, tmp_path):
    run = _write_run(tmp_path, "run", [_row(0), _row(0)])
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


def test_runs_must_have_the_same_seed_set(summarizer, tmp_path):
    first = _write_run(tmp_path, "first", [_row(0), _row(1)])
    second = _write_run(tmp_path, "second", [_row(0), _row(2)])
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([first, second], task="easy")


def test_seed_order_does_not_have_to_match(summarizer, tmp_path):
    first = _write_run(tmp_path, "first", [_row(0), _row(1)])
    second = _write_run(tmp_path, "second", [_row(1), _row(0)])
    report = summarizer.summarize_runs([first, second], task="easy")
    assert report["aggregate"]["episodes"] == 4


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_missing_required_columns_are_rejected(summarizer, tmp_path, column):
    columns = tuple(name for name in REQUIRED_COLUMNS if name != column)
    run = _write_run(tmp_path, "run", [_row()], columns=columns)
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


@pytest.mark.parametrize("change", [
    {"win": "yes"},
    {"death": "2"},
    {"timeout": ""},
    {"seed": "not-an-integer"},
    {"steps": "1.5"},
    {"steps": -1},
    {"hp_lost": -0.1},
    {"damage_dealt": -1},
    {"time_to_kill": -1},
    {"hp_lost": "NaN"},
    {"damage_dealt": "Infinity"},
    {"reward": "-inf"},
    {"reward": ""},
    {"time_to_kill": "NaN"},
    {"win": "True", "death": "True"},
    {"win": "True", "timeout": "True"},
    {"win": "False", "death": "True", "timeout": "True", "time_to_kill": ""},
])
def test_invalid_episode_values_are_rejected(summarizer, tmp_path, change):
    run = _write_run(tmp_path, "run", [_row(**change)])
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


@pytest.mark.parametrize("metrics", [
    {"episodes": 2},
    {"win_rate": 0.5},
    {"death_rate": 1.0},
    {"mean_hp_lost": 20.0},
    {"mean_reward": 0.0},
    {"mean_time_to_kill": None},
    {"zero_damage_death_rate": 1.0},
    {"zero_damage_timeout_rate": 1.0},
    {"win_rate": float("nan")},
])
def test_inconsistent_metrics_file_is_rejected(summarizer, tmp_path, metrics):
    run = _write_run(tmp_path, "run", [_row()], metrics=metrics)
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


def test_matching_metrics_and_roundoff_are_accepted(summarizer, tmp_path):
    run = _write_run(tmp_path, "run", [_row()], metrics={
        "episodes": 1, "win_rate": 1.0, "death_rate": 0.0,
        "timeout_rate": 0.0, "mean_hp_lost": 1.0,
        "mean_reward": 1.0000000000000002, "mean_time_to_kill": 10.0,
    })
    report = summarizer.summarize_runs([run], task="easy")
    assert report["runs"][0]["metrics"]["mean_reward"] == 1.0


@pytest.mark.parametrize("filename", ["metrics.json", "policy.json"])
def test_malformed_json_is_reported_as_summary_error(summarizer, tmp_path, filename):
    run = _write_run(tmp_path, "run", [_row()])
    (run / filename).write_text("{not JSON", encoding="utf-8")
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


def test_no_runs_or_empty_csv_are_rejected(summarizer, tmp_path):
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([], task="easy")
    run = _write_run(tmp_path, "empty", [])
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


def test_missing_csv_is_rejected(summarizer, tmp_path):
    run = tmp_path / "missing_csv"
    run.mkdir()
    with pytest.raises(summarizer.SummaryError):
        summarizer.summarize_runs([run], task="easy")


def test_json_cli_and_script_import_need_no_training_packages(
    monkeypatch, tmp_path, capsys,
):
    run = _write_run(tmp_path, "run", [_row()], metadata=False)
    before = {path.relative_to(tmp_path): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}
    original_import = builtins.__import__
    forbidden = {"torch", "minestudio", "ray", "numpy", "yaml", "omegaconf", "minecombat"}

    def stdlib_only(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"Summary unexpectedly imports {name}")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as isolated:
        isolated.setattr(builtins, "__import__", stdlib_only)
        module = _load_script("minecombat_test_summarize_evals_stdlib")
        status = module.main([str(run), "--task", "easy", "--format", "json"])

    captured = capsys.readouterr()
    assert status == 0
    report = json.loads(captured.out)
    assert report["task"] == "easy"
    assert report["aggregate"]["episodes"] == 1
    assert report["runs"][0]["metadata"]["checkpoint"] is None
    assert "Traceback" not in captured.err
    after = {path.relative_to(tmp_path): path.read_bytes()
             for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_cli_defaults_to_markdown(summarizer, tmp_path, capsys):
    run = _write_run(tmp_path, "example_run", [_row()])
    status = summarizer.main([str(run), "--task", "combined"])
    output = capsys.readouterr().out
    assert status == 0
    assert "combined" in output
    assert "example_run" in output


def test_markdown_shows_all_identity_fields_and_known_values(summarizer, tmp_path):
    checkpoint_hash = "a" * 64
    run = _write_run(tmp_path, "known_identity", [_row()], policy={
        **POLICY, "checkpoint_sha256": checkpoint_hash, "model_revision": "revision-123",
        "tactic_id": 0, "num_tactics": 2, "deterministic": False, "max_steps": 600,
    })
    report = summarizer.summarize_runs([run], task="easy")
    output = summarizer.format_markdown(report)
    identity_lines = [
        line for line in output.splitlines()
        if line.startswith("- ") and line[2:].split(": ", 1)[0] in summarizer.IDENTITY_FIELDS
    ]
    config_hash = hashlib.sha256(CONFIG.encode("utf-8")).hexdigest()
    assert identity_lines == [
        f'- policy: "{POLICY["policy"]}"',
        f'- model: "{POLICY["model"]}"',
        f'- checkpoint: "{POLICY["checkpoint"]}"',
        f'- checkpoint_sha256: "{checkpoint_hash}"',
        '- model_revision: "revision-123"',
        '- tactic_id: 0',
        '- num_tactics: 2',
        '- deterministic: false',
        '- max_steps: 600',
        f'- config_sha256: "{config_hash}"',
        '- git_commit: "578ad888"',
    ]
    assert [line[2:].split(": ", 1)[0] for line in identity_lines] == list(summarizer.IDENTITY_FIELDS)


def test_markdown_shows_all_unknown_identity_fields(summarizer, tmp_path):
    run = _write_run(tmp_path, "unknown_identity", [_row()], metadata=False)
    report = summarizer.summarize_runs([run], task="easy")
    lines = summarizer.format_markdown(report).splitlines()
    for field in summarizer.IDENTITY_FIELDS:
        assert lines.count(f"- {field}: unknown") == 1


def test_markdown_distinguishes_explicit_null_checkpoint_from_unknown(summarizer, tmp_path):
    run = _write_run(tmp_path, "no_checkpoint", [_row()], policy={
        "checkpoint": None, "model_revision": None,
    })
    report = summarizer.summarize_runs([run], task="easy")
    lines = summarizer.format_markdown(report).splitlines()
    assert "- checkpoint: null" in lines
    assert "- checkpoint: unknown" not in lines
    assert "- model_revision: unknown" in lines


def test_cli_reports_invalid_input_without_success_json(summarizer, tmp_path, capsys):
    run = _write_run(tmp_path, "run", [_row(win="invalid")])
    status = summarizer.main([str(run), "--task", "easy", "--format", "json"])
    captured = capsys.readouterr()
    assert status != 0
    assert captured.err
    assert not captured.out
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("task,run_names,rates,expected_mean", [
    ("combined", (
        "20261106_135858_eval_vpt_ckpt",
        "20261106_143436_eval_vpt_ckpt",
        "20261106_150837_eval_vpt_ckpt",
    ), [0.58, 0.66, 0.62], 0.62),
    ("random", (
        "20261106_142035_eval_vpt_ckpt",
        "20261106_145425_eval_vpt_ckpt",
        "20261106_152754_eval_vpt_ckpt",
    ), [0.06, 0.10, 0.14], 0.10),
])
def test_committed_2m_evaluations_match_report(summarizer, task, run_names, rates, expected_mean):
    paths = [ROOT / "experiments" / "eval_2m_4090d" / name for name in run_names]
    report = summarizer.summarize_runs(paths, task=task)
    assert report["aggregate"]["runs"] == 3
    assert report["aggregate"]["episodes"] == 150
    assert report["aggregate"]["unique_seeds"] == 50
    assert [run["metrics"]["win_rate"] for run in report["runs"]] == pytest.approx(
        rates,
    )
    win_stats = report["aggregate"]["per_run"]["win_rate"]
    assert {key: win_stats[key] for key in ("mean", "min", "max")} == pytest.approx({
        "mean": expected_mean, "min": min(rates), "max": max(rates),
    })
