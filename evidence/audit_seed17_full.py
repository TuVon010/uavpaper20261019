"""Read-only, independent acceptance of the COMPLETE seed17 sensitivity download.

No project imports, evaluation, model selection, or writes under --root.  The
optional --out writes only this audit's JSON.  Exit status 1 means incomplete or
invalid data, including any missing job.  Baseline best/last report views are
aliases of the same evaluation, not two independent evaluations.

Training log_frames=False: training CSVs save full-scenario hashes only.
All saved evaluation frame hashes are checked; no unavailable raw training
trajectories are claimed to have been checked. No project modules are imported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


METHODS = (
    "baseline_priority_greedy", "baseline_greedy_solver", "pure_ppo", "hybrid_ppo",
    "hybrid_sac", "hybrid_ddpg", "hybrid_ppo_ar_adaptive", "hybrid_ppo_ar05",
)
SOURCE = "3ef6b15c31a7de84f7ce1fb64ee1e70f71b252e8ee7db35e05766d1db35ca870"
TAG = "seed17_sensitivity_20261020_01"
ERRORS = (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError)
METRICS = (
    "weighted_delay_mean", "mean_delay_s", "total_energy_j", "deadline_completion",
    "episode_return", "budget_feasible", "total_propulsion_energy_j",
    "total_compute_energy_j", "service_disabled_uav_frame_fraction", "total_decision_mean_s",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    actual, expected = number(actual), number(expected)
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
            f"{label}: {actual!r} != {expected!r}")


def number(value):
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return float(value.lower() == "true")
    value = float(value)
    require(math.isfinite(value), "Non-finite metric")
    return value


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def by_episode(records, expected, label):
    require([int(r["episode"]) for r in records] == expected, label + ": incomplete, duplicate, or reordered episodes")
    return {int(r["episode"]): r for r in records}


def vector(value, length, label):
    if isinstance(value, str):
        value = json.loads(value)
    require(isinstance(value, list) and len(value) == length, label + ": invalid vector length")
    return [number(v) for v in value]


def boolean(value):
    require(value in (True, False, 0, 1, "True", "False", "true", "false"),
            "Invalid boolean: " + repr(value))
    return bool(number(value))


def bool_vector(value, length, label):
    if isinstance(value, str):
        value = json.loads(value)
    require(isinstance(value, list) and len(value) == length, label + ": invalid mask")
    return [boolean(v) for v in value]


def hash_string(value, label):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), label + ": invalid SHA256")
    return value


def power(speed, cfg):
    cubic = .5 * cfg["rotor_drag_ratio"] * cfg["air_density"] * cfg["rotor_solidity"] * cfg["rotor_area_m2"]
    return cfg["rotor_p0_w"] * (1 + 3 * speed * speed / cfg["rotor_tip_mps"] ** 2) + cfg["rotor_pi_w"] + cubic * speed ** 3


def audit_evaluation(out, folder, plan, cfg, status, training):
    path = out / folder
    payload = read_json(path / "summary.json")
    summary, records, scenarios = payload["summary"], payload["episodes"], payload["scenarios"]
    protocol = plan["suite"]["evaluation"]
    ids = list(range(protocol["episode_start"], protocol["episode_start"] + protocol["episodes"]))
    records_by_ep = by_episode(records, ids, folder + "/summary.episodes")
    by_episode(scenarios, ids, folder + "/summary.scenarios")
    csv_by_ep = by_episode(read_csv(path / "episodes.csv"), ids, folder + "/episodes.csv")
    for key, wanted in dict(seed=17, episodes=10, episode_start=30000,
                            environment_config=cfg, source=plan["source"], deterministic_policy=True).items():
        require(summary.get(key) == wanted, folder + ": metadata mismatch " + key)
    if training:
        selected = status["best_step"] if folder == "evaluation_best" else 60000
        require(summary.get("selected_step") == selected, folder + ": selected checkpoint differs from existing validation selection")
    frames, users, uavs = cfg["episode_frames"], cfg["num_users"], cfg["num_uavs"]
    grouped = defaultdict(list)
    raw_external_digest = hashlib.sha256()
    with (path / "trajectories.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            require(bool(line.strip()), folder + ": blank trajectory line")
            row = json.loads(line)
            grouped[int(row["episode"])].append(row)
    require(sorted(grouped) == ids, folder + ": trajectory episode set differs")
    for scenario in scenarios:
        episode = int(scenario["episode"])
        record, csv_record, trajectory = records_by_ep[episode], csv_by_ep[episode], grouped[episode]
        label = f"{folder}/episode{episode}"
        hash_string(scenario["scenario_hash"], label)
        require(record["scenario_hash"] == scenario["scenario_hash"] == csv_record["scenario_hash"], label + ": episode hash mismatch")
        require([int(r["frame"]) for r in trajectory] == list(range(frames)), label + ": incomplete/duplicate/reordered frames")
        require(len(scenario["frame_scenario_hashes"]) == frames, label + ": incomplete scenario frame hashes")
        require([r["frame_scenario_hash"] for r in trajectory] == scenario["frame_scenario_hashes"], label + ": trajectory/scenario frame hash mismatch")
        sums = defaultdict(float)
        energy_per_uav = [0.] * uavs
        for row in trajectory:
            require(row["scenario_hash"] == scenario["scenario_hash"], label + ": trajectory episode hash mismatch")
            hash_string(row["frame_scenario_hash"], label)
            require(row["num_users"] == users, label + ": task count mismatch")
            delays = vector(row["delay_s"], users, label + "/delay")
            deadlines = vector(row["deadline_s"], users, label + "/deadline")
            data = vector(row["data_bit"], users, label + "/data")
            raw_external_digest.update(signature(dict(episode=episode, frame=int(row["frame"]), data=data,
                deadline=deadlines, position=row["user_position"],
                signed_speed=row["user_signed_speed_mps"])).encode())
            association = vector(row["association"], users, label + "/association")
            invalid = bool_vector(row["allocation_invalid_mask"], users, label + "/invalid")
            require(all(d >= 0 for d in delays), label + ": negative effective delay")
            require(all(cfg["deadline_min_s"] <= d <= cfg["deadline_max_s"] for d in deadlines), label + ": deadline outside configured range")
            require(all(cfg["data_min_bit"] <= d <= cfg["data_max_bit"] for d in data), label + ": data outside configured range")
            require(all(a == int(a) and 0 <= a <= uavs + 1 for a in association), label + ": invalid association")
            j = sum(t / d for t, d in zip(delays, deadlines)) / users
            seconds = sum(delays) / users
            completion = sum(t <= d + cfg["feasibility_tol"] and not bad for t, d, bad in zip(delays, deadlines, invalid)) / users
            late = sum(max(t / d - 1, 0) for t, d in zip(delays, deadlines)) / users
            close(row["normalized_weighted_delay"], j, label + "/frame J from T/d")
            close(row["weighted_delay_sum"], users * j, label + "/frame weighted sum")
            close(row["mean_delay_s"], seconds, label + "/frame mean delay")
            close(row["deadline_completion"], completion, label + "/completion from delay/deadline")
            close(row["deadline_violation_cost"], late, label + "/late cost from delay/deadline")
            close(row["allocation_invalid_rate"], sum(invalid) / users, label + "/invalid rate")
            energy = vector(row["energy_j"], uavs, label + "/energy")
            propulsion = vector(row["propulsion_energy_j"], uavs, label + "/propulsion")
            compute = vector(row["compute_energy_j"], uavs, label + "/compute")
            disabled = bool_vector(row["uav_service_disabled"], uavs, label + "/disabled")
            require(len(row["uav_velocity"]) == uavs, label + ": missing UAV velocity")
            before = sum(max(e / cfg["energy_budget_j"] - 1, 0) for e in energy_per_uav) / uavs
            for index in range(uavs):
                velocity = vector(row["uav_velocity"][index], 2, label + "/velocity")
                close(propulsion[index], power(math.hypot(*velocity), cfg) * cfg["frame_s"], label + "/propulsion from actual speed")
                expected_compute = cfg["uav_joule_per_cycle"] * cfg["cycles_per_bit"] * sum(d for a, d in zip(association, data) if a == index + 1)
                close(compute[index], expected_compute, label + "/compute from assigned cycles")
                close(energy[index], propulsion[index] + compute[index], label + "/energy decomposition")
                require(disabled[index] == (cfg["uav_service_cutoff_enabled"] and energy_per_uav[index] >= cfg["energy_budget_j"]),
                        label + ": incorrect slot-start service cutoff")
                if disabled[index]:
                    require(compute[index] == 0 and index + 1 not in association, label + ": task admitted after cutoff")
                energy_per_uav[index] += energy[index]
            cumulative = vector(row["cumulative_energy_j"], uavs, label + "/cumulative energy")
            for actual, expected in zip(cumulative, energy_per_uav):
                close(actual, expected, label + "/energy accumulation")
            close(row["total_energy_j"], sum(energy), label + "/frame total energy")
            after = sum(max(e / cfg["energy_budget_j"] - 1, 0) for e in energy_per_uav) / uavs
            energy_cost = cfg["energy_penalty"] * frames * (after - before)
            penalty = (cfg["deadline_penalty"] * late + energy_cost +
                       cfg["collision_penalty"] * boolean(row["collision_attempt"]) +
                       cfg["boundary_penalty"] * boolean(row["boundary_attempt"]) +
                       cfg["allocation_failure_penalty"] * sum(invalid) / users)
            close(row["energy_violation_cost"], energy_cost, label + "/energy penalty")
            close(row["penalty_cost"], penalty, label + "/total penalty")
            close(row["reward"], -cfg["delay_cost_weight"] * j - penalty, label + "/reward from raw inputs")
            sums["weighted_delay_mean"] += j / frames
            sums["mean_delay_s"] += seconds / frames
            sums["total_energy_j"] += sum(energy)
            sums["total_propulsion_energy_j"] += sum(propulsion)
            sums["total_compute_energy_j"] += sum(compute)
            sums["deadline_completion"] += completion / frames
            sums["episode_return"] += number(row["reward"])
            sums["service_disabled_uav_frame_fraction"] += sum(disabled) / (uavs * frames)
            sums["total_decision_mean_s"] += number(row["total_decision_wall_s"]) / frames
        sums["budget_feasible"] = float(all(x <= cfg["energy_budget_j"] + cfg["feasibility_tol"] for x in energy_per_uav))
        require(int(record["frames"]) == frames == int(csv_record["frames"]), label + ": recorded frame count mismatch")
        for metric in METRICS:
            close(record[metric], sums[metric], label + "/summary episode " + metric)
            close(csv_record[metric], sums[metric], label + "/CSV episode " + metric)
        logged_per_uav = vector(record["per_uav_energy_j"], uavs, label + "/episode UAV energy")
        for actual, expected in zip(logged_per_uav, energy_per_uav):
            close(actual, expected, label + "/episode UAV energy")
    for metric in METRICS:
        close(summary[metric], sum(number(r[metric]) for r in records) / len(ids), folder + "/mean " + metric)
    require(summary["energy_feasible_episodes"] == sum(boolean(r["budget_feasible"]) for r in records), folder + ": feasible episode count mismatch")
    payload["_independent_raw_external_sha256"] = raw_external_digest.hexdigest()
    return payload


def audit_job(root, plan, job, cfg):
    out = root / job["run_dir"]
    require((out / "job.done").is_file(), "Missing job.done")
    require(not (out / "job.failed").exists(), "job.failed present")
    status, exp = read_json(out / "status.json"), read_json(out / "experiment.json")
    require(status.get("state") == "complete" and status.get("training_complete") is True and status.get("evaluation_complete") is True,
            "Done marker without both completed training and evaluation")
    require(not status.get("error") and not status.get("signal"), "Status records error/interruption")
    recipe = plan["recipes"][job["method"]]
    require(exp["id"] == job["method"] and exp["environment"] == cfg and exp["steps_target"] == 60000 and not exp.get("smoke"), "Saved experiment mismatch")
    require(read_json(out / "effective_environment.json") == cfg, "Effective environment mismatch")
    require(read_json(out / "effective_training.json") == exp["train_config"], "Effective training mismatch")
    for key in ("kind", "control", "training"):
        require(exp.get(key) == recipe.get(key), "Recipe mismatch " + key)
    require(exp.get("options", {}) == recipe.get("options", {}), "Schedule option mismatch")
    common = read_json(root / "generated/common.json")
    require(exp["protocol"] == common["evaluation"], "Evaluation protocol mismatch")
    require(read_json(out / "effective_protocol.json") == common["evaluation"], "Effective protocol mismatch")
    wanted = dict(common["training"], **recipe.get("training", {}))
    if recipe.get("allow_schedules", False):
        flags = recipe.get("options", {})
        wanted.update(ppo_schedule_steps=60000 if any(flags.values()) else 0,
                      ppo_noise_final_scale=.1 if flags.get("noise_decay", False) else 1.,
                      ppo_actor_lr_final_scale=.2 if flags.get("lr_decay", False) else 1.)
    for key, value in wanted.items():
        require(key == "device" or exp["train_config"].get(key) == value, "Effective training config mismatch " + key)
    training = job["kind"] == "train"
    rows = []
    if training:
        manifest = read_json(out / "manifest.json")
        require(manifest["source"] == plan["source"] and manifest["steps_target"] == 60000, "Training source/budget mismatch")
        require(manifest["physical_seed"] == 17 and manifest["resume"] is None, "Training must start fresh at seed17")
        require(status["step"] == 60000 and status["completed_episodes"] == 1000, "Incomplete training budget")
        rows = read_csv(out / "episodes.csv")
        require(len(rows) == 1000 and sorted(int(r["episode"]) for r in rows) == list(range(1000)), "Training must contain exactly episodes 0..999")
        require(all(int(r["frames"]) == 60 and int(r["num_users"]) == cfg["num_users"] for r in rows), "Training frame/user count mismatch")
        for row in rows:
            label = job["id"] + "/train" + row["episode"]
            hash_string(row["scenario_hash"], label)
            require(0 < int(row["start_step"]) <= int(row["end_step"]) <= 60000, label + ": invalid step interval")
            close(row["weighted_delay_sum_per_frame"], number(row["weighted_delay_mean"]) * cfg["num_users"], label + "/weighted sum")
            close(row["delay_reward_cost_sum"], number(row["weighted_delay_mean"]) * cfg["delay_cost_weight"] * 60, label + "/delay cost sum")
            close(row["delay_reward_sum"], -number(row["delay_reward_cost_sum"]), label + "/negative delay reward")
            close(row["delay_reward_cost_mean"], number(row["weighted_delay_mean"]) * cfg["delay_cost_weight"], label + "/delay cost mean")
            close(row["episode_return"], -number(row["delay_reward_cost_sum"]) - number(row["penalty_sum"]), label + "/return")
            energy = vector(row["per_uav_energy_j"], 3, label + "/energy")
            propulsion = vector(row["per_uav_propulsion_energy_j"], 3, label + "/propulsion")
            compute = vector(row["per_uav_compute_energy_j"], 3, label + "/compute")
            for i in range(3):
                close(energy[i], propulsion[i] + compute[i], label + "/energy decomposition")
            for field, values in (("total_energy_j", energy), ("total_propulsion_energy_j", propulsion), ("total_compute_energy_j", compute)):
                close(row[field], sum(values), label + "/" + field)
            require(boolean(row["budget_feasible"]) == all(e <= cfg["energy_budget_j"] + cfg["feasibility_tol"] for e in energy), label + ": budget flag mismatch")
            require(0 <= number(row["deadline_completion"]) <= 1 and number(row["mean_delay_s"]) >= 0, label + ": invalid QoS")
        for name in ("best.pt", "last.pt", "updates.jsonl", "validation.csv"):
            require((out / name).is_file() and (out / name).stat().st_size > 0, "Missing/empty " + name)
        require(read_json(out / "job.done") == dict(evaluation_best="evaluation_best/summary.json", evaluation_last="evaluation_last/summary.json", training_steps=60000),
                "Training done contents mismatch")
        validation = read_csv(out / "validation.csv")
        require([int(r["step"]) for r in validation] == list(range(3000, 60001, 3000)), "Validation cadence/budget mismatch")
        selected = [int(r["step"]) for r in validation if boolean(r["best_selected"])]
        require(selected and selected[-1] == status["best_step"], "Best checkpoint differs from original validation choices")
    else:
        require(read_json(out / "job.done") == dict(evaluation="evaluation/summary.json"), "Baseline done contents mismatch")
    folders = ("evaluation_best", "evaluation_last") if training else ("evaluation",)
    outputs = {folder: audit_evaluation(out, folder, plan, cfg, status, training) for folder in folders}
    return outputs, rows


def report_checks(root, plan, outputs, training):
    report_root = root / "reports"
    expected_counts = dict(case_summary=288, all_episodes=2520, single_selected_scene17=252,
                           main_training=6000, sweep_best=160, sweep_last=160)
    tables = {name: read_csv(report_root / (name + ".csv")) for name in expected_counts}
    for name, expected in expected_counts.items():
        require(len(tables[name]) == expected, f"reports/{name}.csv: expected {expected} main rows, got {len(tables[name])}")
    for name in ("case_summary", "all_episodes", "single_selected_scene17"):
        keys = [(r["case_id"], r["method"], r["checkpoint"], r.get("episode")) for r in tables[name]]
        require(len(set(keys)) == len(keys), "Duplicate keys in report " + name)
        expected_keys = set()
        for (case, method), payloads in outputs.items():
            checkpoints = ("best", "last") if name == "case_summary" or "evaluation" not in payloads else ("best",)
            for checkpoint in checkpoints:
                ids = (None,) if name == "case_summary" else (30000,) if name == "single_selected_scene17" else range(30000, 30010)
                expected_keys.update((case, method, checkpoint, None if ep is None else str(ep)) for ep in ids)
        require(set(keys) == expected_keys, "Missing/unexpected report keys: " + name)
        for row in tables[name]:
            require(int(row["seed"]) == 17, "Report seed mismatch")
            method, kind = row["method"], row["checkpoint"]
            payloads = outputs[(row["case_id"], method)]
            payload = payloads["evaluation"] if "evaluation" in payloads else payloads["evaluation_" + kind]
            expected = payload["summary"] if name == "case_summary" else next(r for r in payload["episodes"] if int(r["episode"]) == int(row["episode"]))
            if name == "single_selected_scene17":
                require(int(row["episode"]) == 30000, "Selected scene must remain episode30000")
            for metric in METRICS:
                close(row[metric], expected[metric], "reports/" + name + "/" + method + "/" + metric)
            close(row["total_energy_kj"], number(expected["total_energy_j"]) / 1000, "Report kJ conversion")
            if name == "case_summary":
                require(int(row["episodes"]) == 10, "Report evaluation count mismatch")
                require(int(row["energy_feasible_episodes"]) == payload["summary"]["energy_feasible_episodes"], "Report feasible episode count mismatch")
                require(row["selected_step"] == str(expected.get("selected_step", "")), "Report checkpoint selection mismatch")
            else:
                require(row["scenario_hash"] == expected["scenario_hash"], "Report episode hash mismatch")
    means = {(r["case_id"], r["method"], r["checkpoint"]): r for r in tables["case_summary"]}
    for checkpoint in ("best", "last"):
        rows = tables["sweep_" + checkpoint]
        keys = [(r["axis"], number(r["value"]), r["method"]) for r in rows]
        wanted_keys = {(p["axis"], p["value"], m) for p in plan["sweep_points"] for m in METHODS}
        require(len(set(keys)) == len(rows) and set(keys) == wanted_keys, "Sweep report key mismatch: " + checkpoint)
        for row in rows:
            point = next(p for p in plan["sweep_points"] if p["axis"] == row["axis"] and p["value"] == number(row["value"]))
            require(row["case_id"] == point["case_id"] and row["unit"] == point["unit"], "Sweep case/unit mismatch")
            original = means[(row["case_id"], row["method"], checkpoint)]
            require(all(row[k] == v for k, v in original.items()), "Sweep row differs from case summary")
    indexed = {(method, int(row["episode"])): row for (case, method), rows in training.items() if case == "main" for row in rows}
    require(len(indexed) == 6000, "Invalid training episode cardinality")
    report_keys = [(row["method"], int(row["episode"])) for row in tables["main_training"]]
    require(len(set(report_keys)) == 6000 and set(report_keys) == set(indexed), "Consolidated training episodes mismatch")
    for row in tables["main_training"]:
        original = indexed[(row["method"], int(row["episode"]))]
        require(all(row.get(k) == v for k, v in original.items()), "Consolidated training row differs from original")
    require((report_root / "REPORT.md").is_file(), "Missing REPORT.md")
    return expected_counts


def audit_plan(root):
    plan = read_json(root / "suite_plan.json")
    signed = dict(plan)
    require(signed.pop("plan_sha256") == signature(signed), "Plan fingerprint mismatch")
    require(plan["tag"] == TAG and plan["source"]["sha256"] == SOURCE, "Unexpected suite/source")
    suite = plan["suite"]
    require(suite["seed"] == 17 and suite["episodes"] == 1000 and plan["steps_target"] == 60000,
            "Unexpected suite seed or budget")
    require(suite["methods"] == list(METHODS), "Unexpected methods/order")
    require(suite["evaluation"] == dict(episodes=10, episode_start=30000, policy_seed=17, also_sample=False),
            "Unexpected evaluation protocol")
    require(suite["validation"] == dict(episodes=3, episode_start=20000), "Unexpected validation protocol")
    require(suite["data_distribution"] == dict(kind="relative_uniform", lower_fraction=.4, upper_fraction=1.6, unit="Mbit"),
            "Unexpected data-size distribution")
    for relative, expected in plan["generated_hashes"].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Generated path escapes root")
        require(sha(path) == expected, "Generated config hash mismatch: " + relative)
    common = read_json(root / "generated/common.json")
    require(common["episodes"] == 1000 and common["training"]["seed"] == 17, "Generated seed/budget mismatch")
    require(common["evaluation"] == suite["evaluation"], "Generated evaluation protocol mismatch")
    require(common["training"]["eval_start"] == 20000 and common["training"]["eval_episodes"] == 3,
            "Generated validation protocol mismatch")
    cases = {c["id"]: c for c in plan["cases"]}
    require(len(cases) == len(plan["cases"]) == 18, "Expected 18 unique cases")
    main = cases["main"]["environment"]
    require(main["num_users"] == 20 and main["num_uavs"] == 3 and main["episode_frames"] == 60,
            "Unexpected main dimensions")
    require(main["data_min_bit"] == 500000 and main["data_max_bit"] == 2000000, "Main must retain original 0.5--2 Mbit")
    for c in cases.values():
        cfg = c["environment"]
        path = (root / c["environment_file"]).resolve()
        require(path.is_relative_to(root) and read_json(path) == cfg, "Generated case config mismatch")
        changed = {k for k in main.keys() | cfg.keys() if main.get(k) != cfg.get(k)}
        require(changed == set(c["changed_fields"]), "Misreported changed fields: " + c["id"])
        axis, value = c["axis"], c["value"]
        if axis == "main":
            require(c["id"] == "main" and not changed and value is None, "Invalid main")
        elif axis == "users":
            require(changed == {"num_users"} and cfg["num_users"] == value, "User scan changes other factors")
        elif axis == "mean_data_mbit":
            require(changed == {"data_min_bit", "data_max_bit"}, "Data scan changes other factors")
            close(cfg["data_min_bit"], .4 * value * 1e6, "Data lower bound")
            close(cfg["data_max_bit"], 1.6 * value * 1e6, "Data upper bound")
        elif axis in ("uav_cpu_ghz", "uav_bandwidth_mhz"):
            key, scale = ("uav_cpu_hz", 1e9) if axis == "uav_cpu_ghz" else ("uav_bandwidth_hz", 1e6)
            require(changed == {key}, "CPU/BW scan changes other factors")
            close(cfg[key], value * scale, "CPU/BW scan value")
        else:
            raise ValueError("Unexpected axis " + axis)
    axes = dict(users=[10, 15, 20, 25, 30], mean_data_mbit=[.5, 1, 1.5, 2, 2.5],
                uav_cpu_ghz=[5, 10, 15, 20, 25], uav_bandwidth_mhz=[5, 10, 15, 20, 25])
    points = plan["sweep_points"]
    require(len(points) == len({(p["axis"], p["value"]) for p in points}) == 20, "Wrong sweep point set")
    require({a: [p["value"] for p in points if p["axis"] == a] for a in axes} == axes, "Wrong sweep values")
    require(sum(p["case_id"] == "main" for p in points) == 3, "Main must be reused for users20/CPU10/BW10")
    for p in points:
        c = cases[p["case_id"]]
        if c["id"] != "main":
            require(c["axis"] == p["axis"] and c["value"] == p["value"], "Point/case mismatch")
        else:
            require((p["axis"], p["value"]) in (("users", 20), ("uav_cpu_ghz", 10), ("uav_bandwidth_mhz", 10)),
                    "Wrong main alias")
    jobs = plan["jobs"]
    require(len(jobs) == len({j["id"] for j in jobs}) == 144, "Expected 144 unique jobs")
    require({(j["case_id"], j["method"]) for j in jobs} == {(c, m) for c in cases for m in METHODS},
            "Missing/duplicate case-method combinations")
    require(sum(j["kind"] == "train" for j in jobs) == 108, "Expected 108 fresh training jobs")
    for j in jobs:
        require(j["id"] == j["case_id"] + "/" + j["method"], "Job identity mismatch")
        require(j["kind"] == ("baseline" if j["method"].startswith("baseline_") else "train"), "Job kind mismatch")
        require(j["run_dir"] == f"cases/{j['case_id']}/jobs/{j['method']}/{j['method']}", "Unexpected run directory")
    return plan, cases


def suite_completion(root, plan):
    done = read_json(root / "suite.done")
    require(done["plan_sha256"] == plan["plan_sha256"] and done["source_sha256"] == SOURCE and done["jobs"] == 144,
            "suite.done plan/source/job count mismatch")
    require(not (root / "suite.incomplete").exists(), "suite.incomplete present")
    state = read_json(root / "suite_state.json")
    require(state["state"] == "complete" and state["plan_sha256"] == plan["plan_sha256"], "Suite state not complete")
    ids = {j["id"] for j in plan["jobs"]}
    require(set(state["jobs"]) == ids, "Suite state job set mismatch")
    for job_id, item in state["jobs"].items():
        require(item["state"] == "complete" and item.get("returncode") == 0, "Non-successful job: " + job_id)
    collected = read_json(root / "reports/collection_audit.json")
    require(collected["complete"] is True and collected["complete_jobs"] == collected["total_jobs"] == 144 and not collected["errors"],
            "Collection audit is incomplete or records errors")
    require(len(collected["completed"]) == 144 and set(collected["completed"]) == ids, "Collection completed set mismatch")


def audit(root):
    root = root.resolve()
    result = dict(schema_version=1, updated_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                  scope="Complete frozen seed17 suite: 18 configurations / 144 jobs", passed=False,
                  verified_jobs=[], errors=[], summary=[], case_pairings=[],
                  limitations=["Training log_frames=False: only whole-scenario hashes are available for training; no raw training frame trajectories are claimed to have been inspected.",
                               "Models are checked for existence and original validation selection; no deserialization, evaluation, or reselection is performed.",
                               "Saved scene hashes are compared, not regenerated; no project runtime code is imported."])
    try:
        plan, cases = audit_plan(root)
    except ERRORS as error:
        result["errors"].append({"scope": "plan", "error": str(error)})
        return result
    try:
        suite_completion(root, plan)
        result["suite_completion_verified"] = True
    except ERRORS as error:
        result["errors"].append({"scope": "suite_completion", "error": str(error)})
    outputs, training = {}, {}
    for job in plan["jobs"]:
        try:
            cfg = cases[job["case_id"]]["environment"]
            out, rows = audit_job(root, plan, job, cfg)
            key = job["case_id"], job["method"]
            outputs[key] = out
            if rows:
                training[key] = rows
            result["verified_jobs"].append(job["id"])
            payload = out.get("evaluation", out.get("evaluation_best"))["summary"]
            result["summary"].append(dict(case_id=job["case_id"], method=job["method"], **{m: payload[m] for m in METRICS}))
        except ERRORS as error:
            result["errors"].append({"scope": job["id"], "error": str(error)})
    train_hashes = {key: {int(r["episode"]): r["scenario_hash"] for r in rows} for key, rows in training.items()}
    for case in cases:
        try:
            require(all((case, method) in outputs for method in METHODS), "Missing/invalid case method")
            reference = outputs[(case, METHODS[0])]["evaluation"]["scenarios"]
            raw_reference = outputs[(case, METHODS[0])]["evaluation"]["_independent_raw_external_sha256"]
            for method in METHODS:
                for folder, payload in outputs[(case, method)].items():
                    require(payload["scenarios"] == reference, f"Unpaired complete episode/frame scenarios: {method}/{folder}")
                    require(payload["_independent_raw_external_sha256"] == raw_reference,
                            f"Unpaired raw task/deadline/mobility inputs: {method}/{folder}")
            maps = [train_hashes[(case, method)] for method in METHODS[2:]]
            require(len(maps) == 6 and all(mapping == maps[0] for mapping in maps[1:]), "Training episodes use different external scenarios")
            result["case_pairings"].append(dict(case_id=case, training_episodes=1000, evaluation_episodes=10, evaluation_frames=600))
        except ERRORS as error:
            result["errors"].append({"scope": "pairing/" + case, "error": str(error)})
    for case, cfg in cases.items():
        if cfg["axis"] in ("uav_cpu_ghz", "uav_bandwidth_mhz"):
            try:
                require(outputs[(case, METHODS[0])]["evaluation"]["scenarios"] == outputs[("main", METHODS[0])]["evaluation"]["scenarios"],
                        "CPU/BW external evaluation scenario/frame hashes differ from main")
                require(outputs[(case, METHODS[0])]["evaluation"]["_independent_raw_external_sha256"] ==
                        outputs[("main", METHODS[0])]["evaluation"]["_independent_raw_external_sha256"],
                        "CPU/BW raw external task/deadline/mobility inputs differ from main")
                require(train_hashes[(case, METHODS[2])] == train_hashes[("main", METHODS[2])],
                        "CPU/BW external training scenario hashes differ from main")
            except ERRORS as error:
                result["errors"].append({"scope": "cross_case/" + case, "error": str(error)})
    result["counts"] = dict(cases=len(cases), verified_jobs=len(outputs), trained_jobs=len(training),
                            training_episode_rows=sum(len(r) for r in training.values()),
                            evaluation_protocols=sum(len(v) for v in outputs.values()),
                            evaluation_episode_rows=sum(len(p["episodes"]) for v in outputs.values() for p in v.values()),
                            evaluation_frame_rows=sum(len(p["episodes"]) * cases[c]["environment"]["episode_frames"]
                                                      for (c, _), v in outputs.items() for p in v.values()))
    if len(outputs) == 144:
        try:
            result["report_row_counts"] = report_checks(root, plan, outputs, training)
            for key, expected in dict(cases=18, verified_jobs=144, trained_jobs=108, training_episode_rows=108000,
                                      evaluation_protocols=252, evaluation_episode_rows=2520, evaluation_frame_rows=151200).items():
                require(result["counts"][key] == expected, "Wrong total cardinality: " + key)
        except ERRORS as error:
            result["errors"].append({"scope": "reports_and_counts", "error": str(error)})
    result["passed"] = len(result["verified_jobs"]) == 144 and len(result["case_pairings"]) == 18 and not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Suite result folder, not the project root")
    parser.add_argument("--out", type=Path, help="Optional audit JSON outside the suite result root")
    args = parser.parse_args()
    if args.out and args.out.resolve().is_relative_to(args.root.resolve()):
        parser.error("--out must be outside --root to preserve read-only result acceptance")
    result = audit(args.root)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
