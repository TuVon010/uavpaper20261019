"""Audit and report a paired 20-scenario-seed evaluation without rerunning policies.

Input: ROOT/<method>/seed_01.json ... seed_20.json, each containing the
unmodified training.evaluate result plus an evaluation_protocol object.
Only these input JSON files are read; checkpoints and project code are untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


GROUPS = (
    "baseline_priority_greedy", "baseline_greedy_solver", "pure_ppo",
    "hybrid_ppo", "hybrid_sac", "hybrid_ddpg",
    "hybrid_ppo_ar_adaptive", "hybrid_ppo_ar05",
)
LABELS = {
    "baseline_priority_greedy": "Priority Greedy",
    "baseline_greedy_solver": "Greedy＋Solver",
    "pure_ppo": "Pure PPO",
    "hybrid_ppo": "Hybrid PPO",
    "hybrid_sac": "Hybrid SAC",
    "hybrid_ddpg": "Hybrid DDPG",
    "hybrid_ppo_ar_adaptive": "Hybrid PPO AR-adaptive",
    "hybrid_ppo_ar05": "Hybrid PPO AR＝0.5",
}
SEEDS = tuple(range(1, 21))
EPISODE = 30000
CORE_METRICS = (
    "weighted_delay_mean", "mean_delay_s", "deadline_completion",
    "total_energy_j", "episode_return", "budget_feasible",
    "service_disabled_uav_frame_fraction", "collision_attempt_frames",
    "boundary_attempt_frames",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return canonical(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]],
              columns: list[str] | None = None) -> None:
    if columns is None:
        first = ["method", "label", "seed", "episode", "checkpoint_step"]
        all_columns = set().union(*(row.keys() for row in rows))
        columns = [k for k in first if k in all_columns]
        columns += sorted(all_columns - set(columns))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: cell(row.get(key)) for key in columns} for row in rows)


def load_and_audit(root: Path) -> tuple[dict[str, list[dict]], dict]:
    records: dict[str, list[dict]] = {group: [] for group in GROUPS}
    cfg_text: str | None = None
    source_hash: str | None = None
    reference_scenarios: dict[int, dict] = {}
    checkpoints: dict[str, dict] = {}
    input_files: list[dict] = []
    common_cfg = None
    for group in GROUPS:
        reference_model = None
        for seed in SEEDS:
            path = root / group / f"seed_{seed:02d}.json"
            require(path.is_file(), f"Missing result: {path}")
            raw_bytes = path.read_bytes()
            result = json.loads(raw_bytes.decode("utf-8-sig"))
            label = f"{group}/seed_{seed:02d}"
            require(isinstance(result, dict), f"{label}: result must be an object")
            protocol = result.get("evaluation_protocol")
            require(isinstance(protocol, dict), f"{label}: evaluation_protocol missing")
            for field in ("method", "seed", "episode", "checkpoint_step",
                          "checkpoint_sha256", "source_sha256"):
                require(field in protocol, f"{label}: protocol.{field} missing")
            require(protocol["method"] == group, f"{label}: method mismatch")
            require(protocol["seed"] == seed, f"{label}: seed mismatch")
            require(protocol["episode"] == EPISODE, f"{label}: episode must be {EPISODE}")
            require(is_sha256(protocol["source_sha256"]), f"{label}: invalid source hash")
            if group.startswith("baseline_"):
                require(protocol["checkpoint_sha256"] is None
                        and protocol["checkpoint_step"] is None,
                        f"{label}: baseline must not claim a learned checkpoint")
            else:
                require(is_sha256(protocol["checkpoint_sha256"]),
                        f"{label}: invalid checkpoint hash")
                require(isinstance(protocol["checkpoint_step"], int)
                        and protocol["checkpoint_step"] >= 0,
                        f"{label}: invalid checkpoint step")
            summary = result.get("summary")
            episodes = result.get("episodes")
            scenarios = result.get("scenarios")
            require(isinstance(summary, dict), f"{label}: summary missing")
            require(isinstance(episodes, list) and len(episodes) == 1,
                    f"{label}: exactly one evaluated episode is required")
            require(isinstance(scenarios, list) and len(scenarios) == 1,
                    f"{label}: exactly one scenario is required")
            episode, scenario = episodes[0], scenarios[0]
            require(summary.get("episodes") == 1, f"{label}: summary episodes mismatch")
            require(summary.get("seed") == seed, f"{label}: summary seed mismatch")
            require(summary.get("episode_start") == EPISODE,
                    f"{label}: summary episode_start mismatch")
            require(episode.get("episode") == scenario.get("episode") == EPISODE,
                    f"{label}: episode identity mismatch")
            cfg = summary.get("environment_config")
            require(isinstance(cfg, dict) and cfg, f"{label}: environment_config missing")
            cfg_now = canonical(cfg)
            if cfg_text is None:
                cfg_text, common_cfg = cfg_now, cfg
            require(cfg_now == cfg_text, f"{label}: environment parameters differ")
            summary_source = summary.get("source", {}).get("sha256")
            require(summary_source == protocol["source_sha256"],
                    f"{label}: summary/protocol source hashes differ")
            if source_hash is None:
                source_hash = summary_source
            require(summary_source == source_hash, f"{label}: source version differs")
            if not group.startswith("baseline_"):
                require(summary.get("selected_step") == protocol["checkpoint_step"],
                        f"{label}: selected step differs from frozen checkpoint")
            model = {key: protocol[key] for key in
                     ("checkpoint_step", "checkpoint_sha256", "source_sha256")}
            if reference_model is None:
                reference_model = model
                checkpoints[group] = model
            require(model == reference_model,
                    f"{label}: model changed between evaluation seeds")
            scenario_hash = scenario.get("scenario_hash")
            hashes = scenario.get("frame_scenario_hashes")
            frames = episode.get("frames")
            require(is_sha256(scenario_hash) and scenario_hash == episode.get("scenario_hash"),
                    f"{label}: episode scenario hash missing or inconsistent")
            require(isinstance(hashes, list) and hashes and all(map(is_sha256, hashes)),
                    f"{label}: invalid frame hashes")
            require(len(hashes) == frames == cfg.get("episode_frames"),
                    f"{label}: frame count mismatch")
            identity = {"episode": EPISODE, "scenario_hash": scenario_hash,
                        "frame_scenario_hashes": hashes}
            if seed not in reference_scenarios:
                reference_scenarios[seed] = identity
            require(identity == reference_scenarios[seed],
                    f"{label}: exogenous scenario differs from paired methods")
            for key in CORE_METRICS:
                value = episode.get(key)
                require(isinstance(value, (float, int, bool))
                        and math.isfinite(float(value)),
                        f"{label}: finite episode metric {key} is required")
                require(key in summary
                        and math.isclose(float(summary[key]), float(value),
                                         rel_tol=1e-12, abs_tol=1e-12),
                        f"{label}: episode/summary metric differs: {key}")
            energy = episode.get("per_uav_energy_j")
            require(isinstance(energy, list) and len(energy) == cfg.get("num_uavs")
                    and all(is_number(x) and math.isfinite(x) for x in energy),
                    f"{label}: per-UAV energy missing or invalid")
            require(math.isclose(sum(energy), episode["total_energy_j"],
                                 rel_tol=1e-10, abs_tol=1e-7),
                    f"{label}: total energy inconsistent with per-UAV energy")
            row = {"method": group, "label": LABELS[group], "seed": seed,
                   **episode, **model,
                   "total_energy_kj": episode["total_energy_j"] / 1000.0,
                   "max_uav_energy_j": max(energy),
                   "environment_config_sha256": digest(cfg),
                   "frame_scenario_hashes": hashes,
                   "evaluation_protocol": protocol}
            records[group].append(row)
            input_files.append({"path": f"{group}/seed_{seed:02d}.json",
                                "sha256": hashlib.sha256(raw_bytes).hexdigest()})
    require(len({v["scenario_hash"] for v in reference_scenarios.values()}) == 20,
            "Different scene seeds produced duplicate scenario hashes")
    audit = {
        "status": "passed", "methods": list(GROUPS), "seeds": list(SEEDS),
        "episode_per_seed": EPISODE, "evaluated_episodes_per_method": 20,
        "total_method_episode_records": 160,
        "identical_environment_config": True,
        "environment_config_sha256": digest(common_cfg),
        "environment_config": common_cfg,
        "source_sha256": source_hash,
        "same_seed_episode_exogenous_hashes_match_all_methods": True,
        "frame_hash_checks": sum(len(x["frame_scenario_hashes"])
                                 for x in reference_scenarios.values()) * len(GROUPS),
        "distinct_scenario_hashes": 20,
        "checkpoints_fixed_across_seeds": checkpoints,
        "paired_scenarios": {str(k): v for k, v in reference_scenarios.items()},
        "input_files": input_files,
        "aggregation": "20 episodes equally weighted; sample standard deviation ddof=1",
        "interpretation": [
            "One fixed trained best checkpoint per learned method, training seed 1.",
            "Seeds 1..20 vary evaluation scenarios, not training repetitions.",
            "Episode 30000 remains a development evaluation, not the reserved final test.",
            "Seed 1 matches the training master seed but episode 30000 is not a reused training episode.",
            "Evaluation outcomes are not used to reselect checkpoints.",
        ],
    }
    return records, audit


def summarize(records: dict[str, list[dict]]) -> list[dict]:
    output = []
    for group, rows in records.items():
        first = rows[0]
        result = {"method": group, "label": LABELS[group], "episodes": len(rows),
                  "checkpoint_step": first["checkpoint_step"],
                  "checkpoint_sha256": first["checkpoint_sha256"],
                  "source_sha256": first["source_sha256"],
                  "std_ddof": 1,
                  "energy_feasible_episodes": sum(bool(r["budget_feasible"]) for r in rows),
                  "deadline_feasible_episodes": sum(bool(r["deadline_feasible"]) for r in rows),
                  "fully_feasible_episodes": sum(bool(r["fully_feasible"]) for r in rows)}
        numeric = sorted(set.intersection(*(set(k for k, v in row.items()
                             if isinstance(v, (float, int, bool))) for row in rows)))
        for metric in numeric:
            if metric in {"seed", "episode", "checkpoint_step"}:
                continue
            values = [float(r[metric]) for r in rows]
            require(all(map(math.isfinite, values)), f"Nonfinite metric: {group}/{metric}")
            for suffix, value in (("mean", statistics.mean(values)),
                                  ("std", statistics.stdev(values)),
                                  ("median", statistics.median(values)),
                                  ("min", min(values)), ("max", max(values))):
                result[f"{metric}_{suffix}"] = value
        output.append(result)
    return output


def report_text(records: dict[str, list[dict]], summaries: list[dict], audit: dict) -> str:
    lines = [
        "# 停服版本：20个场景种子的配对评估", "",
        "每种学习方法固定使用本轮训练已选出的 **best检查点**，只改变评估场景种子1–20；"
        "每个种子运行episode 30000一次，8种方法逐帧使用相同的外生场景。训练种子仍为1，"
        "因此这是**固定模型的20场景泛化评估，不能当作20次独立训练**。不根据本次评估结果重新挑选模型。", "",
        "种子1与训练的主种子相同，但episode编号30000未重用训练episode。"
        "episode 30000仍属于开发评估；预留的最终测试不在本报告中。", "",
        "所有平均值均为20个episode等权平均；“±”为样本标准差（ddof＝1），"
        "不代表独立训练的不确定性。加权时延成本为每槽平均Tᵢ/dᵢ再按时隙平均，"
        "无量纲；平均时延单位为秒。总能量为3架UAV合计，能量可行要求每架分别不超过电池上限。", "",
        "## 20个episode的平均结果", "",
        "| 方法 | 加权时延成本↓ | 平均时延/s↓ | 按期率 | 总能量/kJ | 能量可行 | 回报↑ | 计算停服UAV帧比例 | 碰撞请求帧 | 越界请求帧 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        def ms(metric: str, precision: int = 4, scale: float = 1.0) -> str:
            return (f"{s[metric + '_mean'] * scale:.{precision}f} ± "
                    f"{s[metric + '_std'] * scale:.{precision}f}")
        lines.append(f"| {s['label']} | {ms('weighted_delay_mean')} | {ms('mean_delay_s')} | "
                     f"{ms('deadline_completion', 2, 100)}% | {ms('total_energy_kj', 3)} | "
                     f"{s['energy_feasible_episodes']}/20 | {ms('episode_return', 2)} | "
                     f"{ms('service_disabled_uav_frame_fraction', 2, 100)}% | "
                     f"{ms('collision_attempt_frames', 2)} | {ms('boundary_attempt_frames', 2)} |")
    lines += ["", "计算停服比例的分母是episode时隙数×UAV数量；碰撞和越界列是请求违规的时隙数，"
              "并不表示安全投影后的实际碰撞或实际越界。", "", "## 加权时延成本的场景分布", "",
              "| 方法 | 均值 | 中位数 | 样本标准差 | 最小值 | 最大值 |",
              "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        values = " | ".join(f"{s['weighted_delay_mean_' + suffix]:.6f}"
                            for suffix in ("mean", "median", "std", "min", "max"))
        lines.append(f"| {s['label']} | {values} |")
    lines += ["", "## 各随机种子的逐项结果", ""]
    for group in GROUPS:
        first = records[group][0]
        selection = ("规则基线，无学习检查点" if group.startswith("baseline_")
                     else f"固定best检查点：训练步数{first['checkpoint_step']}，"
                          f"SHA256 `{first['checkpoint_sha256']}`")
        lines += [f"### {LABELS[group]}", "", selection + "。", "",
                  "| 场景种子 | 加权时延成本 | 平均时延/s | 按期率 | 总能量/kJ | 能量可行 | 回报 | 计算停服比例 | 碰撞请求帧 | 越界请求帧 |",
                  "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|"]
        for r in records[group]:
            lines.append(f"| {r['seed']} | {r['weighted_delay_mean']:.6f} | "
                         f"{r['mean_delay_s']:.6f} | {100*r['deadline_completion']:.2f}% | "
                         f"{r['total_energy_kj']:.3f} | {'是' if r['budget_feasible'] else '否'} | "
                         f"{r['episode_return']:.3f} | "
                         f"{100*r['service_disabled_uav_frame_fraction']:.2f}% | "
                         f"{r['collision_attempt_frames']} | {r['boundary_attempt_frames']} |")
        lines.append("")
    lines += ["## 配对与文件说明", "",
              f"- 配对审计通过：160条方法–场景记录，20个不同的场景哈希，"
              f"{audit['frame_hash_checks']}个方法–逐帧哈希检查；环境配置完全相同。",
              "- 每组20个场景使用同一检查点及同一源码，未用本次结果改选best。",
              "- `per_seed_metrics.csv`：160行，逐episode完整指标、检查点和场景标识。",
              "- `summary.csv`：8行；各数值指标的mean/std/median/min/max及可行episode计数。",
              "- `weighted_delay_by_seed.csv`：20个种子按行配对的8方法加权时延成本，末尾mean行为各方法等权均值。",
              "- `pairing_audit.json`：环境参数、逐帧场景哈希、固定检查点与160个输入文件校验值。",
              "- CSV使用UTF-8 BOM编码，能量列以`_j`或`_kj`明确单位。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    records, audit = load_and_audit(root)
    summaries = summarize(records)
    # Write derived artifacts only after every input has passed all checks.
    write_csv(root / "per_seed_metrics.csv",
              [row for group in GROUPS for row in records[group]])
    write_csv(root / "summary.csv", summaries)
    by_seed = [{"seed": seed, **{group: records[group][seed - 1]["weighted_delay_mean"]
                                for group in GROUPS}} for seed in SEEDS]
    by_seed.append({"seed": "mean", **{group: statistics.mean(
        row["weighted_delay_mean"] for row in records[group]) for group in GROUPS}})
    write_csv(root / "weighted_delay_by_seed.csv", by_seed, ["seed", *GROUPS])
    (root / "pairing_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (root / "REPORT.md").write_text(report_text(records, summaries, audit), encoding="utf-8")
    print(json.dumps({"status": "passed", "root": str(root), "records": 160,
                      "reports": ["REPORT.md", "per_seed_metrics.csv", "summary.csv",
                                  "weighted_delay_by_seed.csv", "pairing_audit.json"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
