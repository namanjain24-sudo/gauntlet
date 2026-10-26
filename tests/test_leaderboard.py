import json
from pathlib import Path

import pytest

from gauntlet import cli, leaderboard

GOOD = {"name": "my-detector", "author": "someone", "detector": "examples/keyword_detector.py:score"}


def write(tmp_path: Path, spec: dict, filename: str | None = None) -> Path:
    path = tmp_path / (filename or f"{spec.get('name', 'x')}.json")
    path.write_text(json.dumps(spec))
    return path


def test_load_spec_accepts_a_minimal_and_a_full_submission(tmp_path):
    assert leaderboard.load_spec(write(tmp_path, GOOD))["name"] == "my-detector"
    full = {**GOOD, "url": "https://example.com", "pip": ["torch"], "per_text": True, "notes": "hi"}
    assert leaderboard.load_spec(write(tmp_path, full))["per_text"] is True


@pytest.mark.parametrize("spec, filename, message", [
    ({**GOOD, "name": "Has Caps"}, "Has Caps.json", "lowercase"),
    ({**GOOD, "name": "renamed"}, "other.json", "must be named"),
    ({k: v for k, v in GOOD.items() if k != "author"}, None, "'author' is required"),
    ({**GOOD, "author": "  "}, None, "'author' is required"),
    ({**GOOD, "detector": "no-colon-here"}, None, "must look like"),
    ({**GOOD, "pip": "torch"}, None, "'pip' must be a list"),
    ({**GOOD, "per_text": "yes"}, None, "must be true or false"),
    ({**GOOD, "wat": 1}, None, "unknown field"),
])
def test_load_spec_rejects_bad_submissions_with_actionable_errors(tmp_path, spec, filename, message):
    with pytest.raises(leaderboard.SpecError, match=message):
        leaderboard.load_spec(write(tmp_path, spec, filename))


def test_load_spec_rejects_malformed_json_and_non_objects(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(leaderboard.SpecError, match="not valid JSON"):
        leaderboard.load_spec(bad)
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2]")
    with pytest.raises(leaderboard.SpecError, match="must be a JSON object"):
        leaderboard.load_spec(arr)


def _record(name, recall, asr_end, auc, baseline=False):
    return {
        "spec": {"name": name, "author": "a", "url": "", "baseline": baseline},
        "config": leaderboard.CONFIG, "config_version": leaderboard.CONFIG_VERSION, "ran_at": "2026-01-01",
        "result": {"recall_mean": recall, "worst_family": "indirect", "worst_recall": 0.1,
                   "arena": {"asr_end": asr_end}, "external": {"auc": auc} if auc is not None else None},
    }


def test_score_is_the_equal_thirds_mean_and_clamps_sub_chance_auc():
    assert leaderboard.score(_record("x", 1.0, 0.0, 1.0)["result"]) == pytest.approx(100.0)
    assert leaderboard.score(_record("x", 0.0, 1.0, 0.5)["result"]) == pytest.approx(0.0)
    assert leaderboard.score(_record("x", 0.6, 0.4, 0.75)["result"]) == pytest.approx(100 * (0.6 + 0.6 + 0.5) / 3)
    # an AUC below chance is as bad as chance, not negative; a missing one is treated the same
    assert leaderboard.score(_record("x", 0.0, 1.0, 0.1)["result"]) == 0.0
    assert leaderboard.score(_record("x", 0.0, 1.0, None)["result"]) == 0.0


def test_render_ranks_submissions_and_keeps_baselines_in_their_own_table():
    md = leaderboard.render([
        _record("weak", 0.1, 0.9, 0.55), _record("strong", 0.9, 0.1, 0.9),
        _record("house", 1.0, 0.0, 0.8, baseline=True),
    ])
    submitted, _, reference = md.partition("## Reference defenders")
    ranks = [ln.split("|")[1].strip() + ":" + ln.split("|")[2].split("<")[0].strip()
             for ln in submitted.splitlines() if ln.startswith("| 1 |") or ln.startswith("| 2 |")]
    assert ranks == ["1:strong", "2:weak"], "higher score must rank first"
    assert "house" in reference and "house" not in submitted
    assert "trained on these attack families" in reference


def test_render_says_so_when_nobody_has_submitted_yet():
    assert "yours would be first" in leaderboard.render([_record("house", 1.0, 0.0, 0.8, baseline=True)])


def test_render_handles_a_detector_with_no_real_data_auc():
    assert "n/a" in leaderboard.render([_record("offline", 0.5, 0.5, None)])


def test_cli_check_validates_every_file_and_fails_loudly_on_a_bad_one(tmp_path, capsys):
    write(tmp_path, GOOD)
    cli.main(["leaderboard", "--check", "--dir", str(tmp_path)])
    assert "1 submission file(s) valid" in capsys.readouterr().err
    write(tmp_path, {**GOOD, "name": "broken", "detector": "oops"}, "broken.json")
    with pytest.raises(SystemExit):
        cli.main(["leaderboard", "--check", "--dir", str(tmp_path)])
    assert "must look like" in capsys.readouterr().err


def test_cli_render_rebuilds_the_board_from_result_files(tmp_path):
    (tmp_path / "x.result.json").write_text(json.dumps(_record("x", 0.5, 0.5, 0.7)))
    out = tmp_path / "BOARD.md"
    cli.main(["leaderboard", "--render", "--dir", str(tmp_path), "--out", str(out)])
    assert "| 1 | x" in out.read_text()


def test_cli_run_benchmarks_a_submission_end_to_end(tmp_path, monkeypatch):
    """The real path CI takes, with a tiny arena budget so the test stays fast."""
    monkeypatch.setitem(leaderboard.CONFIG, "generations", 1)
    monkeypatch.setitem(leaderboard.CONFIG, "pop", 8)
    monkeypatch.setitem(leaderboard.CONFIG, "n_intents", 2)
    root = Path(__file__).resolve().parent.parent
    spec = write(tmp_path, {**GOOD, "detector": f"{root / 'examples' / 'keyword_detector.py'}:score"})
    out = tmp_path / "BOARD.md"
    cli.main(["leaderboard", "--run", str(spec), "--dir", str(tmp_path), "--out", str(out)])

    record = json.loads(leaderboard.result_path(spec).read_text())
    assert record["spec"]["name"] == "my-detector"
    assert record["config"]["generations"] == 1, "the result records the config it was measured at"
    assert set(record["result"]["recall_by_family"]) and 0.0 <= record["result"]["recall_mean"] <= 1.0
    assert "my-detector" in out.read_text()


def test_shipped_submissions_are_valid_and_their_results_match_the_current_config():
    """Guards the real submissions/ directory: a stale or hand-edited board fails CI."""
    for path in sorted(leaderboard.SUBMISSIONS.glob("*.json")):
        if path.name.endswith(".result.json"):
            continue
        leaderboard.load_spec(path)
    for record in leaderboard.load_results():
        assert record["config_version"] == leaderboard.CONFIG_VERSION, f"{record['spec']['name']} needs re-measuring"
        assert record["config"] == leaderboard.CONFIG, f"{record['spec']['name']} was measured at a different config"
    assert leaderboard.BOARD.read_text() == leaderboard.render(leaderboard.load_results()), \
        "LEADERBOARD.md is out of date: run `make leaderboard`"


def test_a_broken_detector_fails_loudly_without_blocking_the_rest_of_the_board(tmp_path, capsys):
    """CI runs third-party code: one detector that raises must not cost everyone else their row."""
    monkey = {"generations": 1, "pop": 8, "n_intents": 2}
    saved = {k: leaderboard.CONFIG[k] for k in monkey}
    leaderboard.CONFIG.update(monkey)
    try:
        root = Path(__file__).resolve().parent.parent
        write(tmp_path, {**GOOD, "name": "works",
                         "detector": f"{root / 'examples' / 'keyword_detector.py'}:score"}, "works.json")
        boom = tmp_path / "boom.py"
        boom.write_text("def score(texts):\n    raise RuntimeError('detector exploded')\n")
        write(tmp_path, {**GOOD, "name": "boom", "detector": f"{boom}:score"}, "boom.json")
        out = tmp_path / "BOARD.md"
        with pytest.raises(SystemExit):
            cli.main(["leaderboard", "--run", "--dir", str(tmp_path), "--out", str(out)])
    finally:
        leaderboard.CONFIG.update(saved)

    err = capsys.readouterr().err
    assert "detector exploded" in err and "1 submission(s) failed" in err
    assert (tmp_path / "works.result.json").exists() and not (tmp_path / "boom.result.json").exists()
    assert "works" in out.read_text() and "boom" not in out.read_text()
