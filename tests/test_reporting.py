"""Integrity checks for terminal-episode reports and the read-only dashboard."""
import csv
from functools import partial
from http.server import ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from priority_mec.reporting import (
    _pairing, extract_trajectory, read_csv_rows, render_comparison, render_run_curves,
    render_trajectories, render_training_comparison, rolling_mean,
)
from tools.live_dashboard import DashboardHandler, metrics_snapshot, safe_path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n", encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
                         for row in rows)


def evaluation_fixture(directory, *, baseline=None, scenario="paired", area=1000):
    initial = [[100., 100.], [500., 100.], [900., 100.]]
    users = [[450., 500.], [550., 500.]]
    trajectories = []
    for frame in range(2):
        before = [[p[0], p[1]+frame*10] for p in initial]
        after = [[p[0], p[1]+(frame+1)*10] for p in initial]
        trajectories.append({"episode": 30000, "frame": frame, "uav_position": before,
                             "next_uav_position": after, "uav_xy_m": after,
                             "user_position": users, "user_xy_m": users})
    episode = {"episode": 30000, "scenario_hash": scenario, "frames": 2,
               "episode_return": -3., "weighted_delay_mean": 1., "total_energy_j": 60.,
               "per_uav_energy_j": [20., 20., 20.], "deadline_completion": .9,
               "fully_feasible": False, "total_decision_mean_s": .001}
    summary = {k: v for k, v in episode.items() if isinstance(v, (int, float, bool))}
    summary.update(environment_config={"area_m": area, "episode_frames": 2},
                   seed=1, episodes=1, baseline=baseline, control="pure")
    result = {"summary": summary, "episodes": [episode],
              "scenarios": [{"episode": 30000, "scenario_hash": scenario,
                             "frame_scenario_hashes": ["frame0", "frame1"]}]}
    write_json(directory/"summary.json", result)
    (directory/"trajectories.jsonl").write_text("".join(json.dumps(row)+"\n" for row in trajectories), encoding="utf-8")
    return result, trajectories


def test_csv_ignores_partial_final_append_and_keeps_full_episodes(tmp_path):
    path = tmp_path/"episodes.csv"
    write_csv(path, [{"episode": 4, "episode_return": -2, "per_uav_energy_j": [1, 2, 3]}])
    with path.open("a", encoding="utf-8") as stream:
        stream.write('5,-3,"[1,')
    rows = read_csv_rows(path)
    assert len(rows) == 1
    assert rows[0]["episode"] == 4
    assert rows[0]["per_uav_energy_j"] == [1, 2, 3]
    assert rolling_mean([10., 20., 30.]) == [10., 15., 20.]
    assert rolling_mean(list(range(30)))[-1] == np.mean(list(range(5, 30)))


def test_baseline_versions_pair_only_when_physics_reward_and_realized_scenes_match(tmp_path):
    import copy
    old, _ = evaluation_fixture(tmp_path/'old')
    old['summary']['environment_config'].update(
        greedy_direction_count=72, greedy_speed_levels=5, greedy_refine_rounds=2)
    new = copy.deepcopy(old)
    new['summary']['environment_config'].update(
        greedy_direction_count=16, greedy_speed_levels=4, greedy_refine_rounds=0)
    assert _pairing(old) == _pairing(new)
    for key in ('area_m', 'energy_budget_j', 'collision_penalty', 'satellite_bandwidth_hz'):
        changed = copy.deepcopy(new)
        changed['summary']['environment_config'][key] = -1
        assert _pairing(old) != _pairing(changed)
    for key in ('scenario_hash', 'frame_scenario_hashes'):
        changed = copy.deepcopy(new)
        changed['scenarios'][0][key] = 'different' if key == 'scenario_hash' else ['changed']
        assert _pairing(old) != _pairing(changed)


def test_episode_rendering_and_real_initial_trajectory(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [{"episode": 2-i, "episode_return": -3.-i, "frames": 2,
             "weighted_delay_mean": 1., "total_energy_j": 60.+i,
             "per_uav_energy_j": [20.+i, 20., 20.], "deadline_completion": .9,
             "fully_feasible": False} for i in range(2)]
    write_csv(tmp_path/"episodes.csv", rows)
    write_json(tmp_path/"manifest.json", {"config": {"energy_budget_j": 100}})
    write_csv(tmp_path/"validation.csv", [{"step": 10, "episode_return": -3.,
              "weighted_delay_mean": 1., "total_energy_j": 60., "deadline_completion": .9,
              "feasible_episodes": 0, "mean_delay_s": 1.}])
    metadata = render_run_curves(tmp_path)
    assert metadata["training_episodes"] == 2
    assert metadata["validation_points"] == 1
    assert metadata["sources"] == ["episodes.csv", "validation.csv"]
    assert read_csv_rows(tmp_path/"episodes.csv")[0]["fully_feasible"] is False
    for file in metadata["figures"]:
        assert (tmp_path/file).stat().st_size > 100
    assert "<text" in (tmp_path/"plots/training_curves.svg").read_text(encoding="utf-8")
    assert "Metric unavailable" not in (tmp_path/"plots/training_curves.svg").read_text(encoding="utf-8")
    _, trajectories = evaluation_fixture(tmp_path/"evaluation_best")
    # Include an unrelated next episode; extraction must stop after the first.
    episode, path = extract_trajectory(trajectories+[dict(trajectories[0], episode=30001)])
    assert episode == 30000
    assert path.shape == (3, 3, 2)
    np.testing.assert_array_equal(path[0], trajectories[0]["uav_position"])
    np.testing.assert_array_equal(path[-1], trajectories[-1]["uav_xy_m"])
    result = render_trajectories(tmp_path, "best")
    assert result["points"] == 3 and result["frames"] == 2
    assert result["uavs"] == 3 and result["users"] == 2
    assert len(result["figures"]) == 3
    assert not render_trajectories(tmp_path, "last")["figures"]
    with pytest.raises(ValueError, match="initial"):
        extract_trajectory([{k: v for k, v in trajectories[0].items() if k != "uav_position"}])


def test_comparison_requires_all_declared_groups_and_paired_sources(tmp_path):
    write_json(tmp_path/"experiment_manifest.json", {"groups": [
        {"id": "rl", "label": "RL", "kind": "train", "control": "pure", "run_dir": "rl"},
        {"id": "local", "label": "Local", "kind": "baseline", "run_dir": "local"}]})
    evaluation_fixture(tmp_path/"rl/evaluation_best")
    with pytest.raises(FileNotFoundError, match="local"):
        render_comparison(tmp_path)
    evaluation_fixture(tmp_path/"local/evaluation", baseline="local", scenario="different")
    with pytest.raises(ValueError, match="Unpaired"):
        render_comparison(tmp_path)
    assert not (tmp_path/"comparison").exists()
    evaluation_fixture(tmp_path/"local/evaluation", baseline="local", area=900)
    with pytest.raises(ValueError, match="Unpaired"):
        render_comparison(tmp_path)
    evaluation_fixture(tmp_path/"local/evaluation", baseline="local")
    pytest.importorskip("matplotlib")
    report = render_comparison(tmp_path)
    assert report["paired"] and report["groups"] == 2
    assert report["rows"][1]["training_curve"] == "not applicable"
    assert report["rows"][0]["uav_1_energy_j"] == 20.
    assert all(row["training_seeds"] == 1 for row in report["rows"])
    assert "no error bars" in report["statistics"]
    assert (tmp_path/report["table"]).exists()
    assert len(report["figures"]) == 6  # metric comparison + full trajectory grid
    for file in report["figures"]:
        assert (tmp_path/file).stat().st_size > 100
    baseline_meta = render_run_curves(tmp_path/"local")
    assert not baseline_meta["training_applicable"]
    assert not baseline_meta["figures"]


def test_training_overlay_uses_eight_histories_and_validates_episode_steps(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import priority_mec.reporting as reporting
    groups = []
    base_rows = [{"episode": index, "start_step": [1, 2, 9][index],
                  "end_step": [7, 8, 15][index], "frames": 4, "scenario_hash": "scenario%d" % index,
                  "episode_return": -10.+index, "weighted_delay_mean": 1.,
                  "total_energy_j": 60.+index, "deadline_completion": .9,
                  "fully_feasible": False, "penalty_sum": 6.-index} for index in range(3)]
    for index in range(10):
        identifier = "method%d" % index
        groups.append({"id": identifier, "kind": "baseline" if index < 2 else "train",
                       "control": "hybrid" if index >= 7 else "pure",
                       "label": "Long method label %d with adaptive autoregression and continuation" % index})
        write_csv(tmp_path/identifier/"episodes.csv", base_rows)
        write_json(tmp_path/identifier/"manifest.json", {"config": {"area_m": 1000, "episode_frames": 4}})
    write_json(tmp_path/"experiment_manifest.json", {"groups": groups})
    inspected = []
    save = reporting._save

    def inspect_save(fig, destination, plt):
        if destination.name == "training_comparison":
            inspected.append([list(ax.lines) for ax in fig.axes[:6]])
        return save(fig, destination, plt)

    monkeypatch.setattr(reporting, "_save", inspect_save)
    previous_level = logging.getLogger("fontTools.subset").level
    metadata = render_training_comparison(tmp_path)
    assert logging.getLogger("fontTools.subset").level == previous_level
    assert metadata["groups"] == 8
    assert metadata["excluded_baselines"] == ["method0", "method1"]
    assert metadata["shared_schedule_verified"]
    assert "end_step" in metadata["schedule_columns_checked"]
    assert len(metadata["figures"]) == 3
    assert all(series["x"] == [1, 2, 3] for series in metadata["series"])
    assert all(series["end_steps"] == [7, 8, 15] for series in metadata["series"])
    assert all(len(lines) == 16 for lines in inspected[0])  # eight raw + eight MA25 lines per panel
    np.testing.assert_array_equal(inspected[0][0][1].get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(inspected[0][0][1].get_ydata(), [-10., -9.5, -9.])
    assert len({lines[1].get_color() for lines in [inspected[0][0][i:i+2] for i in range(0, 16, 2)]}) == 8
    assert all("method0/" not in source and "method1/" not in source for source in metadata["sources"])
    modified = [dict(row) for row in base_rows]
    modified[-1]["end_step"] = 16
    write_csv(tmp_path/"method9/episodes.csv", modified)
    with pytest.raises(ValueError, match="episode/step schedules"):
        render_training_comparison(tmp_path)
    write_csv(tmp_path/"method9/episodes.csv", base_rows)
    write_json(tmp_path/"method9/manifest.json", {"config": {"area_m": 900, "episode_frames": 4}})
    with pytest.raises(ValueError, match="environment configurations"):
        render_training_comparison(tmp_path)


@pytest.mark.parametrize("path", ["/../outside.json", "/%2e%2e/outside.json",
                                  "/folder/%2e%2e/../outside.json", "/C:/outside.json",
                                  "/..%5coutside.json", "/%00.json"])
def test_dashboard_rejects_traversal(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_dashboard_serves_existing_data_read_only_and_refreshes(tmp_path):
    write_csv(tmp_path/"episodes.csv", [{"episode": 0, "episode_return": -2.}])
    assert metrics_snapshot(tmp_path)["runs"][0]["completed_episodes"] == 1
    write_csv(tmp_path/"episodes.csv", [{"episode": 0, "episode_return": -2.},
                                       {"episode": 1, "episode_return": -1.}])
    assert metrics_snapshot(tmp_path)["runs"][0]["completed_episodes"] == 2
    (tmp_path/"unsafe.py").write_text("raise RuntimeError('never execute')", encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(DashboardHandler, root=tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:%d" % server.server_port
    try:
        with urlopen(base+"/api/metrics", timeout=3) as response:
            assert response.headers["Cache-Control"].startswith("no-store")
            assert json.load(response)["runs"][0]["completed_episodes"] == 2
        with urlopen(base+"/", timeout=3) as response:
            assert b"setInterval(refresh,5000)" in response.read()
        for path, code in (("/%2e%2e/outside.json", 403), ("/unsafe.py", 404), ("/plots/", 404)):
            with pytest.raises(HTTPError) as error:
                urlopen(base+path, timeout=3)
            assert error.value.code == code
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base+"/api/metrics", data=b"{}", method="POST"), timeout=3)
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_launcher_dashboard_reaches_actual_server_without_torch(tmp_path):
    project = Path(__file__).resolve().parents[1]
    # A fresh interpreter proves the CLI and its real lazy wrapper/server do
    # not import torch. Only the blocking HTTP server object is mocked.
    code = """
import sys
from unittest.mock import patch
from priority_mec.experiments import main
with patch('tools.live_dashboard.ThreadingHTTPServer') as server:
    instance = server.return_value.__enter__.return_value
    instance.server_port = 8765
    main(['dashboard', '--run', sys.argv[1], '--port', '8765'])
    server.assert_called_once()
    assert server.call_args.args[0] == ('127.0.0.1', 8765)
    instance.serve_forever.assert_called_once_with(poll_interval=.5)
assert 'torch' not in sys.modules
print('dashboard route reached the actual server without torch')
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=project,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "actual server without torch" in result.stdout


def test_direct_dashboard_help_needs_no_package_install():
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    # -S also removes installed site-packages, so only this checkout can make
    # the sibling priority_mec package importable to tools/live_dashboard.py.
    result = subprocess.run([sys.executable, "-S", "tools/live_dashboard.py", "--help"],
                            cwd=project, env=environment, capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--run" in result.stdout and "--port" in result.stdout
