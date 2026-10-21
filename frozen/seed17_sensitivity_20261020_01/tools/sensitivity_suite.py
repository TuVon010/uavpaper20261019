"""Immutable main-scene and one-factor experiment queue (no torch import).

The frozen run.py is the sole training entry point. Every method gets a private
experiment_manifest.json parent to avoid concurrent launcher write races.
"""
from __future__ import annotations
import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = 1
METRICS = ("weighted_delay_mean", "mean_delay_s", "total_energy_j", "deadline_completion",
           "episode_return", "budget_feasible", "total_propulsion_energy_j", "total_compute_energy_j",
           "service_disabled_uav_frame_fraction", "total_decision_mean_s")

def now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def temporary_sibling(path):
    # Short unique names avoid Windows MAX_PATH failures in nested run folders.
    return Path(path).with_name(f".{os.getpid():x}{time.time_ns():x}.tmp")

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = temporary_sibling(path)
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_manifest(root):
    root = Path(root)
    hashes = {p.relative_to(root).as_posix(): sha_file(p) for p in sorted((root / "priority_mec").rglob("*.py"))}
    if not hashes:
        raise ValueError("No priority_mec sources in " + str(root))
    return {"sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "files": hashes}

def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")

def checked_inside(root, name):
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Path must stay inside project: " + str(name))
    return candidate

def validate_suite(suite):
    if suite.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported suite schema")
    for key in ("seed", "episodes", "render_every_episodes"):
        value = suite[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == "episodes" else 0):
            raise ValueError("Invalid integer " + key)
    methods = suite["methods"]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be a nonempty unique list")
    if any(not isinstance(method, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", method) for method in methods):
        raise ValueError("Method IDs must be safe single path components")
    for section in ("evaluation", "validation"):
        for key in ("episodes", "episode_start"):
            value = suite[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == "episodes" else 0):
                raise ValueError("Invalid " + section + "." + key)
    dist = suite["data_distribution"]
    if dist["unit"] != "Mbit" or dist["kind"] not in {"relative_uniform", "fixed_value"}:
        raise ValueError("Data distribution uses Mbit and relative_uniform or fixed_value")
    if dist["kind"] == "relative_uniform":
        lo, hi = dist["lower_fraction"], dist["upper_fraction"]
        if not 0 < lo < hi or not math.isclose((lo + hi) / 2, 1):
            raise ValueError("Relative uniform fractions must have mean 1")
    ids = set()
    for axis in suite["axes"]:
        if not isinstance(axis["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", axis["id"]):
            raise ValueError("Axis IDs must be safe single path components")
        if axis["id"] in ids or not axis["values"] or len(axis["values"]) != len(set(axis["values"])):
            raise ValueError("Axis IDs and values must be unique")
        ids.add(axis["id"])
        if axis["field"] not in {"num_users", "mean_data_mbit", "uav_cpu_hz", "uav_bandwidth_hz"}:
            raise ValueError("Unknown sensitivity axis")
        if any(not math.isfinite(v) or v <= 0 for v in axis["values"]):
            raise ValueError("Axis values must be finite and positive")

def build_cases(base, suite):
    cases = [{"id": "main", "axis": "main", "value": None, "changed_fields": [], "environment": dict(base)}]
    seen, points = {signature(base): "main"}, []
    if suite.get("main_only", False):
        return cases, points
    for axis in suite["axes"]:
        for value in axis["values"]:
            cfg, field = dict(base), axis["field"]
            if field == "mean_data_mbit":
                dist = suite["data_distribution"]
                lo, hi = ((1, 1) if dist["kind"] == "fixed_value" else (dist["lower_fraction"], dist["upper_fraction"]))
                cfg.update(data_min_bit=float(value) * lo * 1e6, data_max_bit=float(value) * hi * 1e6)
                allowed = {"data_min_bit", "data_max_bit"}
            else:
                scaled = value * axis.get("scale", 1)
                cfg[field] = int(scaled) if field == "num_users" else float(scaled)
                if field == "num_users" and int(scaled) != scaled:
                    raise ValueError("User counts must be integral")
                allowed = {field}
            changes = sorted(key for key in cfg if cfg[key] != base[key])
            if not set(changes) <= allowed or set(cfg) != set(base):
                raise ValueError("One-factor audit failed")
            digest = signature(cfg)
            case_id = seen.get(digest)
            if case_id is None:
                case_id = axis["id"] + "_" + safe_name(value)
                cases.append({"id": case_id, "axis": axis["id"], "value": value, "changed_fields": changes, "environment": cfg})
                seen[digest] = case_id
            points.append({"axis": axis["id"], "value": value, "unit": axis["unit"], "case_id": case_id})
    return cases, points

def generate(project, suite_path, tag):
    project = Path(project).resolve()
    if Path(tag).name != tag or tag in {".", ".."}:
        raise ValueError("tag must be one folder name")
    suite_path = Path(suite_path)
    suite_path = suite_path.resolve() if suite_path.is_absolute() else checked_inside(project, suite_path)
    suite = read_json(suite_path)
    validate_suite(suite)
    base = read_json(checked_inside(project, suite["environment"]))
    common = read_json(checked_inside(project, suite["common"]))
    cases, points = build_cases(base, suite)
    common["episodes"] = suite["episodes"]
    common["training"].update(seed=suite["seed"], render_every_episodes=suite["render_every_episodes"],
                               eval_episodes=suite["validation"]["episodes"], eval_start=suite["validation"]["episode_start"])
    common["evaluation"].update(suite["evaluation"])
    if suite["episodes"] * base["episode_frames"] % common["training"]["num_envs"]:
        raise ValueError("Training steps must be divisible by vector environment count")
    recipes = {}
    for method in suite["methods"]:
        recipe = read_json(checked_inside(project, "configs/experiments/" + method + ".json"))
        if recipe["id"] != method or recipe["kind"] not in {"train", "baseline"}:
            raise ValueError("Invalid experiment recipe " + method)
        if "seed" in recipe.get("training", {}) and recipe["training"]["seed"] != suite["seed"]:
            raise ValueError("Recipe would override suite seed")
        if recipe["kind"] == "train":
            count = recipe.get("training", {}).get("num_envs", common["training"]["num_envs"])
            if not isinstance(count, int) or count < 1 or suite["episodes"] % count:
                raise ValueError("episodes must be divisible by num_envs to retain every complete episode: " + method)
        recipes[method] = recipe
    root = project / "results" / tag
    if root.exists():
        raise FileExistsError("Immutable plan already exists: " + str(root))
    root.mkdir(parents=True)
    atomic_json(root / "generated/suite.json", suite)
    atomic_json(root / "generated/common.json", common)
    for case in cases:
        case["environment_file"] = "generated/environment/" + case["id"] + ".json"
        atomic_json(root / case["environment_file"], case["environment"])
    for method, recipe in recipes.items():
        atomic_json(root / "generated/experiments" / (method + ".json"), recipe)
    generated = {p.relative_to(root).as_posix(): sha_file(p) for p in sorted((root / "generated").rglob("*.json"))}
    jobs = []
    for case in cases:
        for method in suite["methods"]:
            out = "cases/{0}/jobs/{1}/{1}".format(case["id"], method)
            jobs.append({"id": case["id"] + "/" + method, "case_id": case["id"], "method": method,
                         "kind": recipes[method]["kind"], "run_dir": out})
    plan = dict(schema_version=SCHEMA, tag=tag, created_utc=now(), suite=suite, cases=cases,
                sweep_points=points, recipes=recipes, jobs=jobs, generated_hashes=generated,
                source=source_manifest(project), steps_target=suite["episodes"] * base["episode_frames"],
                policy="Finish all main methods before one-factor cases; fresh training at every unique configuration.")
    plan["plan_sha256"] = signature(plan)
    atomic_json(root / "suite_plan.json", plan)
    atomic_json(root / "suite_state.json", {"state": "generated", "updated_utc": now(), "plan_sha256": plan["plan_sha256"], "jobs": {}})
    collect(root)
    print(json.dumps({"root": str(root), "unique_cases": len(cases), "sweep_points": len(points), "jobs": len(jobs), "source_sha256": plan["source"]["sha256"]}, indent=2))
    return root

def load_plan(root):
    root = Path(root).resolve()
    plan = read_json(root / "suite_plan.json")
    check = dict(plan)
    digest = check.pop("plan_sha256")
    if digest != signature(check):
        raise ValueError("Plan fingerprint changed")
    for filename, expected in plan["generated_hashes"].items():
        if sha_file(root / filename) != expected:
            raise ValueError("Generated config was modified: " + filename)
    return plan

def runtime_verify(runtime, plan):
    runtime = Path(runtime).resolve()
    if not (runtime / "run.py").is_file() or source_manifest(runtime) != plan["source"]:
        raise ValueError("Frozen runtime source differs from plan")
    for method, expected in plan["recipes"].items():
        if read_json(runtime / "configs/experiments" / (method + ".json")) != expected:
            raise ValueError("Frozen recipe differs: " + method)

def eval_kinds(job):
    return ("evaluation",) if job["kind"] == "baseline" else ("evaluation_best", "evaluation_last")

def completion(root, plan, job):
    """Validate config, source, logs and both evals before calling a job complete."""
    out = root / job["run_dir"]
    if not (out / "job.done").exists():
        return None
    if (out / "job.failed").exists():
        raise ValueError("Both done and failed markers: " + job["id"])
    status = read_json(out / "status.json")
    if status.get("state") != "complete" or not status.get("evaluation_complete") or not status.get("training_complete"):
        raise ValueError("Incomplete status despite job.done: " + job["id"])
    case = next(c for c in plan["cases"] if c["id"] == job["case_id"])
    expected_cfg = case["environment"]
    exp = read_json(out / "experiment.json")
    if exp["id"] != job["method"] or exp["environment"] != expected_cfg or exp["steps_target"] != plan["steps_target"] or exp.get("smoke"):
        raise ValueError("Saved experiment differs: " + job["id"])
    recipe = plan["recipes"][job["method"]]
    for key in ("kind", "control", "training"):
        if exp.get(key) != recipe.get(key):
            raise ValueError("Saved recipe differs: " + job["id"])
    if exp.get("options", {}) != recipe.get("options", {}):
        raise ValueError("Saved schedule switches differ: " + job["id"])
    common = read_json(root / "generated/common.json")
    wanted_train = dict(common["training"])
    wanted_train.update(recipe.get("training", {}))
    if recipe.get("allow_schedules", False):
        flags = recipe.get("options", {})
        wanted_train.update(ppo_schedule_steps=plan["steps_target"] if any(flags.values()) else 0,
                            ppo_noise_final_scale=.1 if flags.get("noise_decay", False) else 1.,
                            ppo_actor_lr_final_scale=.2 if flags.get("lr_decay", False) else 1.)
    for key, value in wanted_train.items():
        if key != "device" and exp["train_config"].get(key) != value:
            raise ValueError("Saved training option differs: " + job["id"] + "/" + key)
    protocol = common["evaluation"]
    if exp["protocol"] != protocol:
        raise ValueError("Saved evaluation protocol differs: " + job["id"])
    outputs = {}
    for folder in eval_kinds(job):
        payload = read_json(out / folder / "summary.json")
        summary, records, scenarios = payload["summary"], payload["episodes"], payload["scenarios"]
        expected = dict(seed=plan["suite"]["seed"], episodes=protocol["episodes"], episode_start=protocol["episode_start"],
                        environment_config=expected_cfg, source=plan["source"], deterministic_policy=True)
        if any(summary.get(k) != v for k, v in expected.items()):
            raise ValueError("Evaluation metadata differs: " + job["id"] + "/" + folder)
        ids = list(range(protocol["episode_start"], protocol["episode_start"] + protocol["episodes"]))
        if [r["episode"] for r in records] != ids or [r["episode"] for r in scenarios] != ids:
            raise ValueError("Incomplete evaluation episode schedule: " + job["id"])
        if any(len(s["frame_scenario_hashes"]) != expected_cfg["episode_frames"] for s in scenarios):
            raise ValueError("Incomplete per-frame hashes: " + job["id"])
        if any(r["scenario_hash"] != s["scenario_hash"] for r, s in zip(records, scenarios)):
            raise ValueError("Episode/scenario hashes differ: " + job["id"])
        for metric in METRICS:
            values = [float(r[metric]) for r in records]
            if not all(math.isfinite(v) for v in values) or not math.isclose(sum(values) / len(values), float(summary[metric]), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Metric mean mismatch: " + job["id"] + "/" + metric)
        outputs[folder] = payload
    if job["kind"] == "train":
        manifest = read_json(out / "manifest.json")
        if manifest["source"] != plan["source"] or manifest["steps_target"] != plan["steps_target"]:
            raise ValueError("Training source/budget differs: " + job["id"])
        if status.get("step") != plan["steps_target"] or status.get("completed_episodes") != plan["suite"]["episodes"]:
            raise ValueError("Training stopped below the intended budget: " + job["id"])
        if outputs["evaluation_last"]["summary"].get("selected_step") != plan["steps_target"]:
            raise ValueError("Last evaluation is not the terminal checkpoint: " + job["id"])
        if outputs["evaluation_best"]["summary"].get("selected_step") != status.get("best_step"):
            raise ValueError("Best evaluation differs from validation selection: " + job["id"])
        with (out / "episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != plan["suite"]["episodes"] or sorted(int(r["episode"]) for r in rows) != list(range(plan["suite"]["episodes"])):
            raise ValueError("Missing per-episode training data: " + job["id"])
        for name in ("best.pt", "last.pt"):
            if not (out / name).is_file():
                raise ValueError("Missing checkpoint: " + job["id"])
    return outputs

def atomic_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = temporary_sibling(path)
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)

@contextmanager
def collection_lock(root):
    """Serialize report snapshots across manual collect and live coordinator."""
    handle = (Path(root) / "collection.lock").open("a+b")
    try:
        if os.name == "posix":
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        else:
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if os.name == "posix":
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        else:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()

def collect(root):
    with collection_lock(root):
        return _collect_unlocked(root)

def _collect_unlocked(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    records, case_rows, completed, errors, pairings, main_training = [], [], [], [], {}, []
    for job in plan["jobs"]:
        try:
            outputs = completion(root, plan, job)
            if outputs is None:
                continue
            completed.append(job["id"])
            if job["case_id"] == "main" and job["kind"] == "train":
                with (root / job["run_dir"] / "episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
                    main_training.extend(dict(method=job["method"], **r) for r in csv.DictReader(stream))
            for folder, payload in outputs.items():
                kind = "last" if folder == "evaluation_last" else "best"
                pair, key = payload["scenarios"], job["case_id"]
                if key in pairings and pairings[key] != pair:
                    raise ValueError("Unpaired same-case scenes: " + job["id"] + "/" + kind)
                pairings[key] = pair
                summary = payload["summary"]
                row = dict(case_id=key, method=job["method"], checkpoint=kind, seed=plan["suite"]["seed"],
                           episodes=len(payload["episodes"]), selected_step=summary.get("selected_step", ""), **{k: summary[k] for k in METRICS})
                row["total_energy_kj"] = row["total_energy_j"] / 1000
                row["energy_feasible_episodes"] = summary["energy_feasible_episodes"]
                case_rows.append(row)
                if folder == "evaluation":
                    case_rows.append(dict(row, checkpoint="last"))
                for episode in payload["episodes"]:
                    records.append(dict(case_id=key, method=job["method"], checkpoint=kind, seed=plan["suite"]["seed"],
                                        episode=episode["episode"], scenario_hash=episode["scenario_hash"], **{k: episode[k] for k in METRICS},
                                        total_energy_kj=episode["total_energy_j"] / 1000))
        except (ValueError, KeyError, OSError) as error:
            errors.append({"job": job["id"], "error": str(error)})
    if "main" in pairings:
        for case in plan["cases"]:
            if case["axis"] in {"uav_cpu_ghz", "uav_bandwidth_mhz"} and case["id"] in pairings and pairings[case["id"]] != pairings["main"]:
                errors.append({"job": case["id"], "error": "CPU/BW external scenario hashes differ from main"})
    metrics = [*METRICS, "total_energy_kj"]
    columns = ["case_id", "method", "checkpoint", "seed", "episodes", "selected_step", *metrics, "energy_feasible_episodes"]
    atomic_csv(root / "reports/case_summary.csv", case_rows, columns)
    ep_columns = ["case_id", "method", "checkpoint", "seed", "episode", "scenario_hash", *metrics]
    atomic_csv(root / "reports/all_episodes.csv", records, ep_columns)
    train_columns = ["method"] + sorted({k for r in main_training for k in r if k != "method"})
    atomic_csv(root / "reports/main_training.csv", main_training, train_columns)
    selected = plan["suite"]["evaluation"]["episode_start"]
    atomic_csv(root / "reports/single_selected_scene17.csv", [r for r in records if r["episode"] == selected], ep_columns)
    for kind in ("best", "last"):
        view = []
        for point in plan["sweep_points"]:
            view.extend(dict(row, axis=point["axis"], value=point["value"], unit=point["unit"])
                        for row in case_rows if row["case_id"] == point["case_id"] and row["checkpoint"] == kind)
        atomic_csv(root / ("reports/sweep_" + kind + ".csv"), view, ["axis", "value", "unit", *columns])
    audit = dict(updated_utc=now(), complete_jobs=len(completed), total_jobs=len(plan["jobs"]), completed=completed, errors=errors,
                 complete=len(completed) == len(plan["jobs"]) and not errors,
                 pairing="Same case: episode and every frame hash; CPU/bandwidth sweeps: external hashes equal main.")
    atomic_json(root / "reports/collection_audit.json", audit)
    if audit["complete"]:
        atomic_json(root / "suite.done", dict(plan_sha256=plan["plan_sha256"], source_sha256=plan["source"]["sha256"], completed_utc=now(), jobs=len(completed)))
    lines = ["# Seed 17 主场景与单因素扫描", "", "更新：" + now(), "",
             f"已验证完成 {len(completed)}/{len(plan['jobs'])} 个实验；独立参数配置 {len(plan['cases'])} 套。", "",
             "训练seed与场景主seed均为17，不同episode和时隙任务仍变化。每个学习方法在每套参数下从头训练。", "",
             "weighted_delay_mean为平均T_i/d_i；mean_delay_s为秒数；total_energy_j为每episode三架无人机推进与计算总能耗，再对10个评估episode取均值。", "",
             "best和last分别记录。best只由既定验证集选取，不使用最终评估选模。基线没有选模，best/last视图重复同一基线。", "",
             "首个评估episode为seed17/episode30000，对应此前选中的主场景；single_selected_scene17.csv单独列出它，case_summary.csv为10个episode均值。", "",
             "训练曲线原始数据在cases/<case>/jobs/<method>/<method>/episodes.csv，1000个episode保留1000行；每100个episode画一次图不减少数据采样。", "",
             "本组是单训练种子的参数敏感性实验，不是组件消融或多训练种子置信区间；seed17为先前比较后选出的开发主场景。", "",
             "## 主场景best", "", "| 方法 | 加权时延成本 | 平均时延/s | 总能耗/kJ | 能量可行episode |", "|---|---:|---:|---:|---:|"]
    for row in case_rows:
        if row["case_id"] == "main" and row["checkpoint"] == "best":
            lines.append(f"| {row['method']} | {row['weighted_delay_mean']:.6g} | {row['mean_delay_s']:.6g} | {row['total_energy_kj']:.6g} | {row['energy_feasible_episodes']}/{row['episodes']} |")
    if errors:
        lines += ["", "## 数据核验异常", ""] + ["- " + e["job"] + ": " + e["error"] for e in errors]
    lines += ["", "## 文件", "", "- case_summary.csv：所有参数配置、方法、best/last均值。", "- all_episodes.csv：逐episode指标。", "- main_training.csv：主场景已完成学习方法的每轮训练数据。", "- single_selected_scene17.csv：各配置首个评估episode。", "- sweep_best.csv / sweep_last.csv：四条扫描曲线长表。", "- collection_audit.json：完成与配对核验。", ""]
    report_path = root / "reports/REPORT.md"
    report_temp = temporary_sibling(report_path)
    report_temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(report_temp, report_path)
    return audit

def process_identity(pid):
    """Linux start ticks avoid mistaking a reused PID for the original child."""
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
        return text[text.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None

def memory_sample(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        return {line.split(":")[0]: int(line.split()[1]) * 1024 for line in lines if line.startswith(("VmRSS:", "VmHWM:"))}
    except OSError:
        return {}

def command_for(root, runtime, plan, job, device):
    case = next(c for c in plan["cases"] if c["id"] == job["case_id"])
    return [sys.executable, "-u", str(runtime / "run.py"), "run", "--experiment", job["method"],
            "--environment", str(root / case["environment_file"]), "--common", str(root / "generated/common.json"),
            "--out", str(root / job["run_dir"]), "--tag", plan["tag"], "--device", device]

def validation_verify(path, runtime, plan):
    proof = read_json(path)
    if proof.get("passed") is not True or proof.get("source_sha256") != plan["source"]["sha256"]:
        raise ValueError("Validation gate failed or belongs to another source")
    for name, expected in proof.get("runtime_files", {}).items():
        if sha_file(checked_inside(runtime, name)) != expected:
            raise ValueError("Validated runtime file changed: " + name)
    return sha_file(path)

def run(project, runtime, root, parallel=4, device="cuda", continue_queue=False, plan_only=False, validation=None):
    project, runtime, root = Path(project).resolve(), Path(runtime).resolve(), Path(root).resolve()
    if not project.is_dir() or not root.is_relative_to(project):
        raise ValueError("Suite root must be inside selected project")
    plan = load_plan(root)
    runtime_verify(runtime, plan)
    if parallel < 1:
        raise ValueError("parallel must be positive")
    if plan_only:
        for job in plan["jobs"]:
            print(json.dumps({"id": job["id"], "argv": command_for(root, runtime, plan, job, device)}))
        return
    if not runtime.is_relative_to(project / "frozen"):
        raise ValueError("Formal runtime must stay inside project/frozen")
    validation = Path(validation).resolve() if validation else project / "ops" / plan["tag"] / "validation.json"
    validation_sha = validation_verify(validation, runtime, plan)
    if os.name != "posix":
        raise RuntimeError("Run training queue on Linux; generate/collect work on Windows")
    import fcntl
    lock = (root / "suite.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another coordinator holds suite.lock")
    state_path = root / "suite_state.json"
    state = read_json(state_path)
    if state["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Queue state belongs to another plan")
    if state["state"] != "generated" and not continue_queue:
        raise RuntimeError("Existing queue: inspect evidence, then explicitly use --continue")
    jobs = state.setdefault("jobs", {})
    for job_id, saved in jobs.items():
        pid = saved.get("pid")
        identity = process_identity(pid) if pid else None
        if saved.get("state") == "running" and identity is not None and identity == saved.get("process_start_ticks"):
            raise RuntimeError(f"Child {job_id} still alive (PID {pid}); do not duplicate it")
    stopping, previous_handlers, running = [False], {}, {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    state.update(state="running", coordinator_pid=os.getpid(), runtime=str(runtime), started_utc=state.get("started_utc", now()),
                 parallel=parallel, validation_file=str(validation), validation_sha256=validation_sha)
    def save():
        state["updated_utc"] = now()
        atomic_json(state_path, state)
    for job in plan["jobs"]:
        try:
            if completion(root, plan, job) is not None:
                jobs.setdefault(job["id"], {}).update(state="complete", updated_utc=now())
                continue
            out = root / job["run_dir"]
            if out.exists() and any(out.iterdir()):
                jobs.setdefault(job["id"], {}).update(state="blocked", reason="Incomplete output retained; manual diagnosis required", updated_utc=now())
            elif jobs.get(job["id"], {}).get("state") == "running":
                jobs[job["id"]].update(state="blocked", reason="Recorded child exited without complete output; manual diagnosis required", updated_utc=now())
        except (ValueError, KeyError, OSError) as error:
            jobs[job["id"]] = {"state": "blocked", "reason": str(error), "updated_utc": now()}
    save()
    collect(root)
    try:
        while True:
            for job_id, item in list(running.items()):
                job, proc, stream, started = item
                snap, saved = memory_sample(proc.pid), jobs[job_id]
                saved["peak_parent_rss_bytes"] = max(saved.get("peak_parent_rss_bytes", 0), snap.get("VmHWM", snap.get("VmRSS", 0)))
                code = proc.poll()
                if code is None:
                    continue
                stream.close()
                saved.update(returncode=code, finished_utc=now(), wall_s=time.monotonic() - started)
                try:
                    if code != 0 or completion(root, plan, job) is None:
                        raise ValueError("Process exit %s; complete training/evaluation evidence missing" % code)
                    saved["state"] = "complete"
                    if job["kind"] == "train":
                        status = read_json(root / job["run_dir"] / "status.json")
                        saved["training_steps_per_total_wall_s"] = float(status["step"]) / saved["wall_s"]
                except (ValueError, KeyError, OSError) as error:
                    saved.update(state="failed", reason=str(error))
                del running[job_id]
                save()
                audit = collect(root)
                print(json.dumps({"finished": job_id, "state": saved["state"], "verified_complete_jobs": audit["complete_jobs"]}), flush=True)
            main_jobs = [j for j in plan["jobs"] if j["case_id"] == "main"]
            main_ok = all(jobs.get(j["id"], {}).get("state") == "complete" for j in main_jobs)
            latest_audit = read_json(root / "reports/collection_audit.json")
            main_ok = main_ok and not any(e["job"].startswith("main/") for e in latest_audit["errors"])
            eligible = [j for j in plan["jobs"] if (main_ok or j["case_id"] == "main") and j["id"] not in jobs]
            while eligible and len(running) < parallel and not stopping[0]:
                job = eligible.pop(0)
                out = root / job["run_dir"]
                out.parent.mkdir(parents=True, exist_ok=True)
                argv = command_for(root, runtime, plan, job, device)
                stream = (out.parent / "launcher.log").open("ab", buffering=0)
                started = time.monotonic()
                proc = subprocess.Popen(argv, cwd=runtime, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                jobs[job["id"]] = dict(state="running", pid=proc.pid, process_start_ticks=process_identity(proc.pid), started_utc=now(),
                                       command=argv, launcher_log=(out.parent / "launcher.log").relative_to(root).as_posix())
                running[job["id"]] = (job, proc, stream, started)
                save()
                print(json.dumps({"started": job["id"], "pid": proc.pid}), flush=True)
            if stopping[0]:
                state["state"] = "coordinator_stopped_children_preserved"
                save()
                return
            if not running:
                audit = collect(root)
                state["state"] = "complete" if audit["complete"] else "attention_required"
                state["collection_audit"] = "reports/collection_audit.json"
                if not audit["complete"]:
                    atomic_json(root / "suite.incomplete", dict(plan_sha256=plan["plan_sha256"], checked_utc=now(),
                        completed= audit["complete_jobs"], total_jobs=len(plan["jobs"]), jobs=jobs, audit_errors=audit["errors"]))
                save()
                return
            save()
            time.sleep(5)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for _, _, stream, _ in running.values():
            stream.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Create immutable config/recipe/queue snapshot")
    gen.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    gen.add_argument("--suite", default="configs/suites/seed17_sensitivity.json")
    gen.add_argument("--tag", required=True)
    runner = sub.add_parser("run", help="Linux queue; detached tmux recommended")
    runner.add_argument("--project", type=Path, required=True)
    runner.add_argument("--runtime", type=Path, required=True)
    runner.add_argument("--root", type=Path, required=True)
    runner.add_argument("--parallel", type=int, default=4)
    runner.add_argument("--device", default="cuda")
    runner.add_argument("--continue", dest="continue_queue", action="store_true")
    runner.add_argument("--plan-only", action="store_true")
    runner.add_argument("--validation", type=Path, help="Defaults to project/ops/<tag>/validation.json; must pass for this frozen source")
    coll = sub.add_parser("collect", help="Validate completed runs and refresh partial/final tables")
    coll.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.project, args.suite, args.tag)
    elif args.command == "run":
        run(args.project, args.runtime, args.root, args.parallel, args.device, args.continue_queue, args.plan_only, args.validation)
    else:
        print(json.dumps(collect(args.root), indent=2))

if __name__ == "__main__":
    main()
