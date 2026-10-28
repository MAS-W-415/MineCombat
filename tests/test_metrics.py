from __future__ import annotations

import pytest

from minecombat.eval.metrics import EpisodeRecord, summarize, to_csv


def record(**overrides):
    base = dict(
        seed=0,
        win=True,
        death=False,
        timeout=False,
        steps=40,
        hp_lost=2.0,
        damage_dealt=20.0,
        time_to_kill=40,
        reward=7.5,
    )
    base.update(overrides)
    return EpisodeRecord(**base)


def test_summary_metrics() -> None:
    records = [
        record(seed=0, win=True, steps=40, hp_lost=2.0, time_to_kill=40),
        record(seed=1, win=False, death=True, steps=60, hp_lost=20.0, time_to_kill=None),
        record(seed=2, win=False, timeout=True, steps=600, hp_lost=5.0, time_to_kill=None, damage_dealt=0.0),
    ]
    summary = summarize(records)

    assert summary["episodes"] == 3
    assert summary["win_rate"] == pytest.approx(1 / 3)
    assert summary["death_rate"] == pytest.approx(1 / 3)
    assert summary["timeout_rate"] == pytest.approx(1 / 3)
    assert summary["stuck_rate"] == pytest.approx(1 / 3)
    assert summary["mean_hp_lost"] == pytest.approx(27.0 / 3)
    assert summary["mean_time_to_kill"] == pytest.approx(40.0)
    assert summary["damage_ratio"] == pytest.approx(40.0 / 27.0)


def test_empty_summary_raises() -> None:
    with pytest.raises(ValueError):
        summarize([])


def test_csv_roundtrip(tmp_path) -> None:
    path = to_csv([record()], tmp_path / "episodes.csv")
    text = path.read_text(encoding="utf-8")
    assert "seed,win,death,timeout,steps,hp_lost,damage_dealt,time_to_kill,reward" in text
    assert "0,True,False,False,40,2.0,20.0,40,7.5" in text
