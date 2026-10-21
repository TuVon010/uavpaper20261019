"""Renderer regression tests operate only on fixtures, never archived raw data."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import plot_results as plots
from plot_support import read_csv


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "figures.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["workers"] = 1
    result["_protocol"] = {"training_seed": 17, "evaluation_episodes": list(range(30000, 30010))}
    return result


def stub_case_readers(monkeypatch):
    """Read contracts are separately tested against real archived results."""
    monkeypatch.setattr(plots, "run_dir", lambda root, case, method: Path(root) / case / method["id"])
    monkeypatch.setattr(plots, "read_json", lambda path: {"energy_budget_j": 12000, "area_m": 1000})
    monkeypatch.setattr(plots, "training_records", lambda run: [{"episode_return": "-1", "episode": "0", "end_step": "60"}])
    monkeypatch.setattr(plots, "evaluation_records", lambda run, cp: {"summary": {}, "episodes": []})


def test_case_checkpoint_last_never_draws_selected_best(monkeypatch, tmp_path, config):
    stub_case_readers(monkeypatch)
    config["performance"]["checkpoints"] = ["last"]
    calls = []
    def record_bar(root, out, case, methods, runs, cfg, view, checkpoint, selected=False):
        calls.append((view, checkpoint, selected))
    monkeypatch.setattr(plots, "bar_figure", record_bar)
    plots.case_worker((str(tmp_path), "main", config, ["hybrid_ppo"], ["performance"], "test"))
    assert calls
    assert all(cp == "last" for _, cp, _ in calls)
    assert any(selected for _, _, selected in calls)


def test_per_run_and_trajectory_checkpoint_union(monkeypatch, tmp_path, config):
    stub_case_readers(monkeypatch)
    config["trajectories"]["checkpoints"] = ["best"]
    config["per_run"]["checkpoint"] = "last"
    loaded = []
    def get_frames(run, checkpoint, episode):
        loaded.append(checkpoint)
        return [{"frame": checkpoint}]
    monkeypatch.setattr(plots, "trajectory_frames", get_frames)
    def overview(root, case, method, data, cfg, tag):
        assert data["frames"]["last"] == [{"frame": "last"}]
    monkeypatch.setattr(plots, "overview", overview)
    def trajectories(root, out, case, methods, runs, cfg, cp):
        assert cp == "best"
        assert runs["hybrid_ppo"]["frames"][cp] == [{"frame": "best"}]
    monkeypatch.setattr(plots, "trajectory_figure", trajectories)
    plots.case_worker((str(tmp_path), "main", config, ["hybrid_ppo"], ["trajectories", "per_run"], "test"))
    assert set(loaded) == {"best", "last"}


def test_disabled_ar05_removed_from_all_case_outputs(monkeypatch, tmp_path, config):
    stub_case_readers(monkeypatch)
    next(m for m in config["methods"] if m["id"] == "hybrid_ppo_ar05")["enabled"] = False
    visited = []
    def training(root, out, case, methods, *args):
        visited.extend(m["id"] for m in methods)
    monkeypatch.setattr(plots, "train_figure", training)
    monkeypatch.setattr(plots, "bar_figure", training)
    monkeypatch.setattr(plots, "trajectory_figure", training)
    monkeypatch.setattr(plots, "trajectory_frames", lambda *args: [])
    def overview(root, case, method, *args):
        visited.append(method["id"])
    monkeypatch.setattr(plots, "overview", overview)
    plots.case_worker((str(tmp_path), "main", config, None,
                       ["training", "performance", "trajectories", "per_run"], "test"))
    assert visited
    assert "hybrid_ppo_ar05" not in visited
    assert "hybrid_ppo_ar_adaptive" in visited


def fixture_project(tmp_path, config, seed=29):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    root = project / "results" / "fixture_suite"
    root.mkdir(parents=True)
    config["suite"] = "results/fixture_suite"
    (project / "configs" / "figures.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "suite_plan.json").write_text(json.dumps({
        "cases": [{"id": "main"}], "suite": {"seed": seed,
        "evaluation": {"episode_start": 70000, "episodes": 10}}}), encoding="utf-8")
    return project, root


def test_cli_last_selection_and_protocol_derive_from_plan(monkeypatch, tmp_path, config):
    project, root = fixture_project(tmp_path, config)
    # Preserve the real config/plan reads and exact metadata output; mock plotting.
    captured = []
    def case_worker(payload):
        _, case, effective, requested, groups, tag = payload
        captured.append(copy.deepcopy(effective))
        assert case == "main" and requested == ["hybrid_ppo"]
        assert groups == ["performance"]
        for section in ("performance", "sweeps", "trajectories"):
            assert effective[section]["checkpoints"] == ["last"]
        assert effective["per_run"]["checkpoint"] == "last"
        return []
    monkeypatch.setattr(plots, "case_worker", case_worker)
    assert plots.main(["--project", str(project), "--methods", "hybrid_ppo", "--cases", "main",
                       "--groups", "performance", "--checkpoint", "last", "--output-tag", "fixture"]) == 0
    manifest = json.loads((root / "figures" / "fixture" / "manifest.json").read_text())
    assert captured and manifest["training_seed"] == 29
    assert manifest["evaluation_episodes"] == list(range(70000, 70010))
    assert manifest["methods"] == ["hybrid_ppo"]
    assert "seed (29)" in (root / "figures" / "fixture" / "index.html").read_text()


def test_cli_rejects_unknown_and_disabled_methods_before_drawing(monkeypatch, tmp_path, config):
    next(m for m in config["methods"] if m["id"] == "hybrid_ppo_ar05")["enabled"] = False
    project, root = fixture_project(tmp_path, config)
    def should_not_draw(payload):
        pytest.fail("Invalid algorithm selection must be rejected before plotting")
    monkeypatch.setattr(plots, "case_worker", should_not_draw)
    for method in ["hybrid_ppo_ar05", "typo"]:
        with pytest.raises(ValueError, match="Unknown or disabled"):
            plots.main(["--project", str(project), "--methods", method, "--groups", "training"])
    assert not (root / "figures").exists()


def test_training_source_csv_preserves_raw_long_tail(monkeypatch, tmp_path, config):
    method = next(m for m in config["methods"] if m["id"] == "hybrid_ppo")
    values = [-1, -3, -1_000_000, -2]
    training = [{"episode_return": str(value), "episode": str(i), "end_step": str((i + 1) * 60)}
                for i, value in enumerate(values)]
    config["training"]["smoothing_window"] = 2
    config["formats"] = []  # Only the genuine source CSV is written in this test.
    result = plots.train_figure(tmp_path, tmp_path / "case_figures", "main", [method],
                               {method["id"]: {"training": training}}, config, "all")
    rows = read_csv(tmp_path / result["source_csv"])
    assert [float(row["raw_return"]) for row in rows] == values
    assert [float(row["trailing_mean"]) for row in rows] == [-1, -2, -500001.5, -500001]
    assert [int(row["completed_episode"]) for row in rows] == [1, 2, 3, 4]
    assert [row["original_episode_id"] for row in rows] == ["0", "1", "2", "3"]
    assert "No clipping" in result["note"] and "confidence intervals" in result["note"]


def test_sweep_axes_include_late_methods_and_long_tails(monkeypatch, tmp_path, config):
    """A later, much larger policy must expand every final y limit.

    Calling set_ylim(bottom=0) after the first policy disables autoscaling; the
    regression then hides the later policy even though its CSV is still intact.
    """
    methods = [copy.deepcopy(config["methods"][0]), copy.deepcopy(config["methods"][2])]
    config["methods"] = methods
    config["views"] = {"all": {"label": "All methods", "methods": None}}
    config["sweeps"]["checkpoints"] = ["best"]
    rows = []
    for axis in plots.AXES:
        for method_index, method in enumerate(methods):
            for value in [1, 2, 3, 4, 5]:
                # The second method is deliberately four orders larger.
                magnitude = (1 if method_index == 0 else 10000) * value
                rows.append({"axis": axis, "method": method["id"], "value": str(value),
                             "weighted_delay_mean": str(magnitude),
                             "mean_delay_s": str(2 * magnitude),
                             "total_energy_j": str(1000 * magnitude)})
    monkeypatch.setattr(plots, "read_csv", lambda path: rows)
    checked = []
    def inspect_plot(fig, base, source, root, cfg, title, note, kind, **extra):
        assert len(source) == 10  # Both methods, all five points remain present.
        assert kind == "sweep"
        for axis in fig.axes:
            low, high = axis.get_ylim()
            all_points = [float(y) for line in axis.lines for y in line.get_ydata()]
            assert low == 0
            assert max(all_points) <= high
            assert high > 10000  # Cannot pass by only drawing the first method.
            assert len(axis.lines) == 2
        checked.append(extra["axis"])
        plots.plt.close(fig)
        return {"kind": kind}
    monkeypatch.setattr(plots, "save", inspect_plot)
    plots.sweep_figures(tmp_path, config, None, "fixture")
    assert set(checked) == set(plots.AXES)
