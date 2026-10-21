"""Plot data contracts: protect checkpoint identity, long tails, and raw geometry."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from plot_support import (active_methods, evaluation_dir, evaluation_records, num,
                          read_csv, rolling_mean, run_dir, training_records,
                          trajectory_frames, trajectory_positions, view_methods)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mock_run(tmp_path, baseline=False):
    run = tmp_path / "run"
    write_json(run / "experiment.json", {"kind": "baseline" if baseline else "train"})
    for checkpoint, val in [("best", 1.0), ("last", 3.0)]:
        directory = run / ("evaluation" if baseline else f"evaluation_{checkpoint}")
        # One large episode must remain in the ordinary ten-episode mean.
        rows = [{"episode": 30000 + i, "weighted_delay_mean": val if i < 9 else 100.0,
                 "mean_delay_s": 2 * val, "total_energy_j": 30000.0} for i in range(10)]
        summary = {"episodes": 10, "weighted_delay_mean": (9 * val + 100) / 10,
                   "mean_delay_s": 2 * val, "total_energy_j": 30000.0}
        write_json(directory / "summary.json", {"summary": summary})
        write_csv(directory / "episodes.csv", rows)
    return run


def test_exact_checkpoint_identity_and_no_fallback(tmp_path):
    run = mock_run(tmp_path)
    best, last = evaluation_records(run, "best"), evaluation_records(run, "last")
    assert best["summary"]["weighted_delay_mean"] == 10.9
    assert last["summary"]["weighted_delay_mean"] == 12.7
    assert best["directory"] != last["directory"]
    (run / "evaluation_last" / "summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        evaluation_records(run, "last")
    with pytest.raises(ValueError):
        evaluation_dir(run, "latest")


def test_baseline_alias_has_no_fake_training(tmp_path):
    run = mock_run(tmp_path, baseline=True)
    assert evaluation_dir(run, "best") == evaluation_dir(run, "last")
    assert training_records(run) == []


def test_missing_episodes_and_disagreeing_summary_fail(tmp_path):
    run = mock_run(tmp_path)
    directory = run / "evaluation_best"
    summary_path = directory / "summary.json"
    doc = json.loads(summary_path.read_text())
    doc["summary"]["mean_delay_s"] += .1
    write_json(summary_path, doc)
    with pytest.raises(ValueError, match="disagrees"):
        evaluation_records(run, "best")
    rows = read_csv(directory / "episodes.csv")
    write_csv(directory / "episodes.csv", rows[:-1])
    with pytest.raises(ValueError, match="ten"):
        evaluation_records(run, "best")


def test_global_disable_applies_to_every_view():
    config = {"methods": [{"id": "ppo", "enabled": True}, {"id": "ar05", "enabled": False}],
              "views": {"training": {"methods": None}, "bars": {"methods": ["ppo", "ar05"]},
                        "sweeps": {"methods": ["ar05"]}}}
    for view in config["views"]:
        assert "ar05" not in [m["id"] for m in view_methods(config, view)]
    assert active_methods(config, ["ppo"]) == [config["methods"][0]]
    with pytest.raises(ValueError):
        active_methods(config, ["ar05"])
    with pytest.raises(ValueError):
        active_methods(config, ["unknown"])


def test_trailing_smoothing_keeps_initial_sample_and_long_tail():
    assert rolling_mean([1, 3, 11, 1000], 2) == [1, 2, 7, 505.5]
    assert rolling_mean([1, 3], 10) == [1, 2]
    assert rolling_mean([], 2) == []
    with pytest.raises(ValueError):
        rolling_mean([1], 0)
    for value in ["nan", "inf", "-inf"]:
        with pytest.raises(ValueError):
            num(value)
    assert num("False") == 0 and num("True") == 1


def frame_data():
    return [{"episode": 30000, "frame": t, "uav_position": [[t, 10]],
             "uav_xy_m": [[t + 1, 10]], "next_uav_position": [[t + 1, 10]]}
            for t in range(60)]


def test_true_initial_geometry_and_missing_post_point_fail():
    frames = frame_data()
    positions = trajectory_positions(frames)
    assert positions.shape == (61, 1, 2)
    np.testing.assert_array_equal(positions[:, 0, 0], np.arange(61))
    del frames[0]["uav_position"]
    with pytest.raises(KeyError):
        trajectory_positions(frames)
    frames = frame_data()
    del frames[-1]["uav_xy_m"]
    del frames[-1]["next_uav_position"]
    with pytest.raises(KeyError):
        trajectory_positions(frames)


def test_trajectory_sorting_and_duplicates(tmp_path):
    run = mock_run(tmp_path)
    path = run / "evaluation_best" / "trajectories.jsonl"
    frames = frame_data()
    path.write_text("\n".join(json.dumps(row) for row in reversed(frames)), encoding="utf-8")
    assert trajectory_frames(run, "best", 30000) == frames
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n" + json.dumps(frames[0]))
    with pytest.raises(ValueError, match="exactly frames"):
        trajectory_frames(run, "best", 30000)


def test_run_path_requires_unique_planned_job(tmp_path):
    (tmp_path / "a").mkdir()
    write_json(tmp_path / "suite_plan.json", {"jobs": [{"case_id": "main", "method": "m", "run_dir": "a"}]})
    assert run_dir(tmp_path, "main", {"id": "m"}) == (tmp_path / "a").resolve()
    with pytest.raises(ValueError):
        run_dir(tmp_path, "main", "not_planned")


def test_real_archived_main_has_all_checkpoints_and_true_trajectories():
    root = Path(__file__).resolve().parents[1] / "results" / "seed17_sensitivity_20261020_01"
    if not root.is_dir():
        root = Path("E:/postgraduateLife/paper_code/uav_mec_final_1019/results/seed17_sensitivity_20261020_01")
    if not root.is_dir():
        pytest.skip("Archived seed17 data are not installed")
    plan = json.loads((root / "suite_plan.json").read_text(encoding="utf-8"))
    main_jobs = [job for job in plan["jobs"] if job["case_id"] == "main"]
    assert len(main_jobs) == 8
    for job in main_jobs:
        run = run_dir(root, "main", job["method"])
        for checkpoint in ["best", "last"]:
            records = evaluation_records(run, checkpoint)
            assert len(records["episodes"]) == 10
            frames = trajectory_frames(run, checkpoint, 30000)
            assert trajectory_positions(frames, "uav").shape == (61, 3, 2)
            assert trajectory_positions(frames, "user").shape == (61, 20, 2)
        assert len(training_records(run)) == (0 if job["kind"] == "baseline" else 1000)
