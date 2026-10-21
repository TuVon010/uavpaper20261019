"""Coordinator bookkeeping tests; no policy training or torch import."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sensitivity_suite", PROJECT / "tools/sensitivity_suite.py")
SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUITE)


def fixture_project(tmp_path, learned=False):
    root = tmp_path / "project"
    root.mkdir()
    shutil.copytree(PROJECT / "configs", root / "configs")
    common = SUITE.read_json(root / "configs/common.json")
    common["training"]["num_envs"] = 2
    SUITE.atomic_json(root / "configs/common.json", common)
    (root / "priority_mec").mkdir()
    (root / "priority_mec/__init__.py").write_text("# fixture\n")
    (root / "run.py").write_text("# fixture\n")
    cfg = SUITE.read_json(root / "configs/suites/seed17_sensitivity.json")
    cfg.update(main_only=True, episodes=2,
               methods=["baseline_priority_greedy"] + (["hybrid_ppo"] if learned else []))
    cfg["evaluation"]["episodes"] = 2
    SUITE.atomic_json(root / "suite.json", cfg)
    result = SUITE.generate(root, "suite.json", "software_test")
    return root, result, SUITE.load_plan(result)


def fake_complete(root, plan, job, hashes="same"):
    out = root / job["run_dir"]
    out.mkdir(parents=True)
    recipe = plan["recipes"][job["method"]]
    common = SUITE.read_json(root / "generated/common.json")
    train = dict(common["training"], **recipe["training"])
    cfg = plan["cases"][0]["environment"]
    SUITE.atomic_json(out / "experiment.json", dict(recipe, environment=cfg,
        train_config=train, steps_target=plan["steps_target"], protocol=common["evaluation"], options=recipe.get("options", {}), smoke=False))
    SUITE.atomic_json(out / "job.done", {})
    SUITE.atomic_json(out / "status.json", dict(state="complete", evaluation_complete=True,
        training_complete=True, step=plan["steps_target"], best_step=60, completed_episodes=2))
    for folder in SUITE.eval_kinds(job):
        records = [dict(episode=e, scenario_hash=hashes + str(e), **{m: 2.0 for m in SUITE.METRICS}) for e in (30000, 30001)]
        summary = dict(seed=17, episodes=2, episode_start=30000, environment_config=cfg,
            source=plan["source"], deterministic_policy=True, energy_feasible_episodes=2,
            selected_step=plan["steps_target"] if folder == "evaluation_last" else 60, **{m: 2.0 for m in SUITE.METRICS})
        scenes = [dict(episode=r["episode"], scenario_hash=r["scenario_hash"], frame_scenario_hashes=[hashes] * cfg["episode_frames"]) for r in records]
        SUITE.atomic_json(out / folder / "summary.json", dict(summary=summary, episodes=records, scenarios=scenes))
    if job["kind"] == "train":
        SUITE.atomic_json(out / "manifest.json", dict(source=plan["source"], steps_target=plan["steps_target"]))
        (out / "best.pt").write_bytes(b"fixture checkpoint")
        (out / "last.pt").write_bytes(b"fixture checkpoint")
        (out / "episodes.csv").write_text("episode\n0\n1\n", encoding="utf-8")


def test_default_grid_has_18_unique_cases_and_20_scan_points():
    config = SUITE.read_json(PROJECT / "configs/suites/seed17_sensitivity.json")
    base = SUITE.read_json(PROJECT / config["environment"])
    SUITE.validate_suite(config)
    cases, points = SUITE.build_cases(base, config)
    assert len(cases) == 18 and len(points) == 20
    assert sum(p["case_id"] == "main" for p in points) == 3
    assert cases[0]["environment"]["data_min_bit"] == 500000
    for case in cases[1:]:
        assert len(case["changed_fields"]) == (2 if case["axis"] == "mean_data_mbit" else 1)
    sizes = {c["value"]: c["environment"] for c in cases if c["axis"] == "mean_data_mbit"}
    assert sizes[.5]["data_min_bit"] == 200000
    assert sizes[2.5]["data_max_bit"] == 4000000


def test_plan_is_immutable_and_generate_refuses_overwrite(tmp_path):
    project, root, plan = fixture_project(tmp_path)
    assert plan["suite"]["seed"] == 17
    with pytest.raises(FileExistsError):
        SUITE.generate(project, "suite.json", "software_test")
    target = root / "generated/common.json"
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="modified"):
        SUITE.load_plan(root)


def test_collect_partial_and_complete_and_single_scene(tmp_path):
    _, root, plan = fixture_project(tmp_path, learned=True)
    assert SUITE.collect(root)["complete_jobs"] == 0
    for job in plan["jobs"]:
        fake_complete(root, plan, job)
    audit = SUITE.collect(root)
    assert audit["complete"] and audit["complete_jobs"] == 2
    with (root / "reports/case_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(float(r["total_energy_kj"]) == .002 for r in rows)
    with (root / "reports/single_selected_scene17.csv").open(newline="") as stream:
        selected = list(csv.DictReader(stream))
    assert len(selected) == 3 and {r["episode"] for r in selected} == {"30000"}


def test_collect_rejects_pairing_drift(tmp_path):
    _, root, plan = fixture_project(tmp_path, learned=True)
    fake_complete(root, plan, plan["jobs"][0])
    fake_complete(root, plan, plan["jobs"][1], hashes="wrong")
    audit = SUITE.collect(root)
    assert not audit["complete"]
    assert any("Unpaired" in e["error"] for e in audit["errors"])


def test_done_marker_alone_does_not_mean_complete(tmp_path):
    _, root, plan = fixture_project(tmp_path)
    fake_complete(root, plan, plan["jobs"][0])
    out = root / plan["jobs"][0]["run_dir"]
    SUITE.atomic_json(out / "status.json", dict(state="training_complete", training_complete=True))
    audit = SUITE.collect(root)
    assert not audit["complete"] and "Incomplete status" in audit["errors"][0]["error"]


def test_plan_only_has_private_method_parent_and_does_not_create_logs(tmp_path, capsys):
    project, root, plan = fixture_project(tmp_path, learned=True)
    SUITE.run(project, project, root, device="cpu", plan_only=True)
    output = capsys.readouterr().out
    assert "--experiment" in output
    parents = [(root / job["run_dir"]).parent for job in plan["jobs"]]
    assert len(set(parents)) == 2 and not any(p.exists() for p in parents)


def test_validation_gate_requires_correct_passed_source_and_files(tmp_path):
    project, root, plan = fixture_project(tmp_path)
    proof = tmp_path / "validation.json"
    SUITE.atomic_json(proof, dict(passed=False, source_sha256=plan["source"]["sha256"]))
    with pytest.raises(ValueError, match="Validation gate"):
        SUITE.validation_verify(proof, project, plan)
    SUITE.atomic_json(proof, dict(passed=True, source_sha256=plan["source"]["sha256"], runtime_files={"run.py": SUITE.sha_file(project / "run.py")}))
    assert SUITE.validation_verify(proof, project, plan) == SUITE.sha_file(proof)
    (project / "run.py").write_text("# drift\n")
    with pytest.raises(ValueError, match="runtime file changed"):
        SUITE.validation_verify(proof, project, plan)


def test_axis_path_traversal_is_rejected():
    config = SUITE.read_json(PROJECT / "configs/suites/seed17_sensitivity.json")
    config["axes"][0]["id"] = "../outside"
    with pytest.raises(ValueError, match="safe single"):
        SUITE.validate_suite(config)


def test_collect_reports_complete_marker_and_main_training(tmp_path):
    _, root, plan = fixture_project(tmp_path, learned=True)
    for job in plan["jobs"]:
        fake_complete(root, plan, job)
    assert SUITE.collect(root)["complete"]
    assert (root / "suite.done").exists()
    with (root / "reports/main_training.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and {r["method"] for r in rows} == {"hybrid_ppo"}
