"""Reproduce all or part of a saved suite into a new result directory.

Default: inspect only. --prepare creates an immutable plan but starts nothing.
--execute prepares and delegates to the existing validated Linux suite runner.
This tool does not tune parameters, select models or reuse trained checkpoints.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys


def load_runner(project):
    path = project / "tools/sensitivity_suite.py"
    spec = importlib.util.spec_from_file_location("saved_suite_runner", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load existing suite runner: " + str(path))
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def select(values, requested, name):
    if requested is None or requested == ["all"]:
        return list(values)
    if len(requested) != len(set(requested)) or any(x not in values for x in requested):
        raise ValueError("Unknown or duplicate " + name + "; choose from " + ", ".join(values))
    return list(requested)


def prepare_plan(api, original, new_tag, methods, cases):
    plan = deepcopy(original)
    plan.pop("plan_sha256")
    plan.update(tag=new_tag, created_utc=datetime.now(timezone.utc).isoformat())
    plan["suite"]["methods"] = methods
    plan["cases"] = [x for x in plan["cases"] if x["id"] in cases]
    plan["recipes"] = {x: plan["recipes"][x] for x in methods}
    plan["jobs"] = [x for x in plan["jobs"] if x["case_id"] in cases and x["method"] in methods]
    plan["sweep_points"] = [x for x in plan["sweep_points"] if x["case_id"] in cases]
    plan["reproduction_of"] = dict(tag=original["tag"], plan_sha256=original["plan_sha256"],
        methods=methods, cases=cases,
        policy="Fresh training with the original immutable inputs and evaluation protocol; no checkpoint reuse.")
    plan["plan_sha256"] = api.signature(plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--from-suite", default="results/seed17_sensitivity_20261020_01")
    parser.add_argument("--tag", required=True, help="Fresh result folder name; existing folders are never overwritten")
    parser.add_argument("--cases", nargs="+", help="Saved case IDs, or all; default all")
    parser.add_argument("--methods", nargs="+", help="Saved method IDs, or all; default all")
    parser.add_argument("--runtime", type=Path, help="Original frozen runtime; default frozen/<original-tag>")
    parser.add_argument("--validation", type=Path, help="Original saved validation proof")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print plan and commands; this is the default")
    mode.add_argument("--prepare", action="store_true", help="Write only a fresh immutable plan; do not train")
    mode.add_argument("--execute", action="store_true", help="Prepare the plan and start the validated Linux queue")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    if not args.tag or Path(args.tag).name != args.tag or args.tag in {".", ".."}:
        parser.error("--tag must be one folder name")
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    api = load_runner(project)
    source = Path(args.from_suite)
    source = (source if source.is_absolute() else project / source).resolve()
    original = api.load_plan(source)
    methods = select(original["suite"]["methods"], args.methods, "methods")
    cases = select([x["id"] for x in original["cases"]], args.cases, "cases")
    root = (project / "results" / args.tag).resolve()
    if not root.is_relative_to(project):
        raise ValueError("Result directory must stay inside selected project")
    if root.exists():
        raise FileExistsError("Output already exists; use a new --tag, or continue its queue separately: " + str(root))
    runtime = args.runtime or Path("frozen") / original["tag"]
    runtime = (runtime if runtime.is_absolute() else project / runtime).resolve()
    validation = args.validation or Path("ops") / original["tag"] / "validation.json"
    validation = (validation if validation.is_absolute() else project / validation).resolve()
    plan = prepare_plan(api, original, args.tag, methods, cases)
    # All modes check original frozen source/recipes. Dry run deliberately does
    # not need CUDA, torch, Linux, or a validation file, and never writes files.
    api.runtime_verify(runtime, plan)
    commands = [dict(id=job["id"], argv=api.command_for(root, runtime, plan, job, args.device)) for job in plan["jobs"]]
    preview = dict(mode="execute" if args.execute else "prepare" if args.prepare else "dry_run",
        root=str(root), runtime=str(runtime), validation=str(validation),
        methods=methods, cases=cases, jobs=len(commands),
        training_jobs=sum(j["kind"] == "train" for j in plan["jobs"]),
        episodes_per_training_job=plan["suite"]["episodes"], source_sha256=plan["source"]["sha256"],
        command_preview=commands)
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    if not args.prepare and not args.execute:
        return 0
    if args.execute:
        # Check before creating any output. The original runner enforces this
        # again immediately before spawning children.
        if sys.platform != "linux":
            raise RuntimeError("--execute uses the Linux suite runner; use --prepare or --dry-run on Windows")
        if not runtime.is_relative_to(project / "frozen"):
            raise ValueError("Formal runtime must stay inside project/frozen")
        api.validation_verify(validation, runtime, plan)
    root.mkdir(parents=True, exist_ok=False)
    for filename, expected in plan["generated_hashes"].items():
        target = root / filename
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe generated input path: " + filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / filename).read_bytes())
        if api.sha_file(target) != expected:
            raise ValueError("Copied input hash mismatch: " + filename)
    api.atomic_json(root / "suite_plan.json", plan)
    api.atomic_json(root / "suite_state.json", dict(state="generated", updated_utc=api.now(),
        plan_sha256=plan["plan_sha256"], jobs={}))
    api.atomic_json(root / "reproduction_request.json", dict(source=str(source), **preview))
    api.collect(root)
    if args.execute:
        api.run(project, runtime, root, args.parallel, args.device, validation=validation)
    else:
        print("Prepared only. Start the existing runner with --root " + str(root) + " and --runtime " + str(runtime))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
