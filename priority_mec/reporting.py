"""Source-traceable episode plots, paired evaluations and full UAV paths.

This module deliberately has no training, torch, pandas or server dependency.
CSV/JSON are the source of truth. Missing quantities stay missing; a missing
checkpoint evaluation never silently falls back to a different checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import textwrap

ROLLING_WINDOW = 25
FORMATS = ("png", "svg", "pdf")
COLORS = ("#377EB8", "#D28438", "#629B80", "#9A70A7", "#696969")
TRAINING_COLORS = ("#377EB8", "#D28438", "#629B80", "#9A70A7", "#A14B54", "#6F6F6F", "#488C99", "#96723C")


def serve(root: Path, port=8765):
    """CLI entry point; import the read-only HTTP server only on demand."""
    from tools.live_dashboard import serve as serve_dashboard
    return serve_dashboard(Path(root), port=port)


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _decode(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return value
    # csv.DictWriter writes Python bool spelling, whereas JSON arrays use
    # lowercase JSON booleans. Both encodings occur in the native log schema.
    if value in ("True", "False"):
        return value == "True"
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def read_csv_rows(path: Path):
    """Read committed CSV records, ignoring an in-progress final append.

    append_csv always commits a newline. Discarding a final unterminated line
    also handles a quoted list field interrupted halfway through a write.
    """
    try:
        data = Path(path).read_text(encoding="utf-8-sig")
    except (FileNotFoundError, OSError, UnicodeError):
        return []
    if not data.endswith(("\n", "\r")):
        data = data[:data.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(data), strict=True)
    rows = []
    try:
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                continue
            rows.append({k: _decode(v) for k, v in row.items()})
    except csv.Error:
        # Only fully parsed preceding records can participate in a live plot.
        pass
    return rows


def _number(value):
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rolling_mean(values, window=ROLLING_WINDOW):
    """Trailing finite observations, with no fabricated leading zeroes."""
    result = []
    for index in range(len(values)):
        finite = [_number(v) for v in values[max(0, index-window+1):index+1]]
        finite = [v for v in finite if v is not None]
        result.append(sum(finite)/len(finite) if finite else float("nan"))
    return result


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [font for font in ("Microsoft YaHei", "Arial", "DejaVu Sans")
             if font in available]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fonts or ["DejaVu Sans"],
        "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "axes.unicode_minus": False,
        "legend.frameon": False, "savefig.facecolor": "white",
    })
    return plt


def _save(fig, destination: Path, plt):
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    # Microsoft YaHei includes optional MERG/meta tables which fontTools drops
    # during PDF subsetting. Suppress that harmless export chatter only here.
    subset_logger = logging.getLogger("fontTools.subset")
    previous_level = subset_logger.level
    subset_logger.setLevel(logging.ERROR)
    try:
        for extension in FORMATS:
            path = destination.with_suffix("."+extension)
            temporary = path.with_suffix("."+extension+".tmp")
            fig.savefig(temporary, format=extension, dpi=180, bbox_inches="tight")
            os.replace(temporary, path)
            paths.append(path)
    finally:
        subset_logger.setLevel(previous_level)
        plt.close(fig)
    return paths


def _relative(paths, root):
    return [Path(p).relative_to(root).as_posix() for p in paths]


def _config(run_dir):
    for filename in ("environment_config.json", "config.json"):
        value = _json(run_dir/filename)
        if isinstance(value, dict):
            return value.get("environment_config", value.get("config", value))
    for filename in ("manifest.json", "run_manifest.json", "run_config.json", "status.json"):
        value = _json(run_dir/filename, {})
        if isinstance(value, dict):
            for key in ("environment_config", "config", "physics"):
                if isinstance(value.get(key), dict):
                    return value[key]
    return {}


def _line(ax, x, y, label, color=COLORS[0], smooth=True):
    import numpy as np
    y = np.asarray([_number(v) if _number(v) is not None else np.nan for v in y])
    ax.plot(x, y, color=color, alpha=.23 if smooth else 1., linewidth=.7 if smooth else 1.6,
            marker=None if smooth else "o", markersize=3,
            label=None if smooth else label)
    if smooth:
        ax.plot(x, rolling_mean(list(y)), color=color, linewidth=1.6,
                marker="o" if len(x) == 1 else None, markersize=3,
                label=label+" (trailing 25)")
    ax.grid(alpha=.18)


def _penalty(row):
    direct = _number(row.get("penalty_sum"))
    if direct is not None:
        return direct
    reward, frames, delay = (_number(row.get(key)) for key in
                             ("episode_return", "frames", "weighted_delay_mean"))
    return -reward-frames*delay if all(v is not None for v in (reward, frames, delay)) else None


def _training_rows(run_dir):
    rows = read_csv_rows(run_dir/"episodes.csv")
    # Evaluation-only/baseline episodes are not learning histories.
    if (run_dir/"summary.json").exists() and not any(
            (run_dir/name).exists() for name in ("train_config.json", "validation.csv", "train.csv")):
        return []
    rows = [row for row in rows if _number(row.get("episode_return")) is not None]
    # An episode is appended once at termination, in completion order. Do not
    # re-sort vector episodes by their IDs or re-expand into frame rewards.
    return rows


def _baseline_run(run_dir):
    for path in (run_dir/"summary.json", run_dir/"evaluation"/"summary.json"):
        result = _json(path, {})
        if isinstance(result, dict) and result.get("summary", result).get("baseline"):
            return True
    return False


def render_run_curves(run_dir: Path, *, checkpoint_kind=None):
    """Refresh training/validation figures and optionally checkpoint paths.

    Returns JSON-serializable metadata. No plots are invented before the first
    completed episode; absent panels explicitly say that a metric is unavailable.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = _training_rows(root)
    validation = read_csv_rows(root/"validation.csv")
    metadata = {"run_dir": str(root), "training_episodes": len(rows),
                "validation_points": len(validation), "rolling_window": ROLLING_WINDOW,
                "training_applicable": not _baseline_run(root),
                "sources": [], "figures": [], "warnings": []}
    plt = None
    if rows:
        plt = _pyplot()
        x = list(range(1, len(rows)+1))
        fig, axes = plt.subplots(3, 2, figsize=(10.2, 9.1), constrained_layout=True)
        panels = [("episode_return", "Episode return (sum of frame rewards)"),
                  ("weighted_delay_mean", "J: mean normalized weighted delay"),
                  ("total_energy_j", "Energy per episode (J)"),
                  ("deadline_completion", "Deadline completion fraction"),
                  ("penalty_sum", "Total penalty per episode"),
                  ("fully_feasible", "Fully feasible episode (0/1)")]
        for index, ((key, ylabel), ax) in enumerate(zip(panels, axes.flat)):
            values = [_penalty(row) if key == "penalty_sum" else row.get(key) for row in rows]
            if key == "total_energy_j":
                values = [v if v is not None else sum(r["per_uav_energy_j"])
                          if isinstance(r.get("per_uav_energy_j"), list) else None
                          for v, r in zip(values, rows)]
            if any(_number(v) is not None for v in values):
                _line(ax, x, values, "Completed episode")
                if key in ("deadline_completion", "fully_feasible"):
                    ax.set_ylim(-.025, 1.025)
            else:
                ax.text(.5, .5, "Metric unavailable", ha="center", transform=ax.transAxes)
            ax.set_xlabel("Completed episode (completion order)")
            ax.set_ylabel(ylabel)
            ax.set_title(chr(97+index), loc="left", fontweight="bold")
        axes.flat[0].legend(loc="best", fontsize=8)
        fig.suptitle("Training: raw episodes and trailing 25-episode mean", fontsize=11)
        metadata["figures"] += _relative(_save(fig, root/"plots"/"training_curves", plt), root)
        metadata["sources"].append("episodes.csv")
        vectors = [r.get("per_uav_energy_j") for r in rows]
        n_uavs = max((len(v) for v in vectors if isinstance(v, list)), default=0)
        if n_uavs:
            fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
            for uav in range(n_uavs):
                _line(ax, x, [v[uav] if isinstance(v, list) and len(v)>uav else None for v in vectors],
                      "UAV %d" % (uav+1), COLORS[uav % len(COLORS)])
            budget = _number(_config(root).get("energy_budget_j"))
            if budget is not None:
                ax.axhline(budget, color="#555555", linestyle="--", linewidth=1, label="Per-UAV budget")
            ax.set(xlabel="Completed episode (completion order)", ylabel="UAV episode energy (J)")
            ax.legend(ncol=2, fontsize=8)
            metadata["figures"] += _relative(_save(fig, root/"plots"/"training_energy", plt), root)
    else:
        metadata["warnings"].append("No completed training episodes; baseline training curves are not applicable."
                                    if not metadata["training_applicable"] else "Waiting for completed episodes.csv records.")
    if validation:
        plt = plt or _pyplot()
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), constrained_layout=True)
        x = [_number(r.get("step")) for r in validation]
        panels = [("episode_return", "Validation episode return"),
                  ("weighted_delay_mean", "Validation J"), ("total_energy_j", "Validation energy (J)"),
                  ("deadline_completion", "Validation completion fraction"),
                  ("feasible_episodes", "Fully feasible validation episodes"),
                  ("mean_delay_s", "Validation mean delay (s)")]
        for (key, ylabel), ax in zip(panels, axes.flat):
            _line(ax, x, [r.get(key) for r in validation], "Validation", smooth=False)
            ax.set(xlabel="Environment training steps", ylabel=ylabel)
        metadata["figures"] += _relative(_save(fig, root/"plots"/"validation_curves", plt), root)
        metadata["sources"].append("validation.csv")
    if checkpoint_kind is not None:
        path_meta = render_trajectories(root, checkpoint_kind=checkpoint_kind)
        metadata["figures"] += path_meta["figures"]
        metadata["warnings"] += path_meta.get("warnings", [])
    _write_json(root/"plots"/"report_metadata.json", metadata)
    return metadata


def _evaluation_dir(run_dir, checkpoint_kind, baseline=False):
    if checkpoint_kind not in ("best", "last", "final"):
        raise ValueError("checkpoint_kind must be 'best' or 'last'")
    checkpoint_kind = "last" if checkpoint_kind == "final" else checkpoint_kind
    if baseline:
        for directory in (run_dir/"evaluation", run_dir):
            if (directory/"summary.json").is_file():
                return directory
        return run_dir/"evaluation"
    if (run_dir/("evaluation_"+checkpoint_kind)/"summary.json").is_file():
        return run_dir/("evaluation_"+checkpoint_kind)
    if (run_dir/"summary.json").is_file():
        return run_dir
    if (run_dir/"evaluation"/"summary.json").is_file():
        return run_dir/"evaluation"
    return run_dir/("evaluation_"+checkpoint_kind)


def _trajectory_rows(path):
    try:
        stream = path.open(encoding="utf-8-sig")
    except OSError:
        return []
    rows = []
    with stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def extract_trajectory(rows, entity="uav"):
    """Return q0..qH for the first logged evaluation episode.

    The pre-step position in row zero is q0. The post-step field of each row
    supplies q1..qH. Never prepend a separately guessed default position.
    """
    import numpy as np
    if not rows:
        raise ValueError("No trajectory records")
    episode = rows[0].get("episode")
    selected = [r for r in rows if r.get("episode") == episode]
    before = selected[0].get(entity+"_position")
    if before is None:
        raise ValueError("Trajectory lacks the recorded initial %s_position q0" % entity)
    positions = [before]
    for row in selected:
        after = row.get(entity+"_xy_m", row.get("next_"+entity+"_position"))
        if after is None:
            raise ValueError("Trajectory lacks post-step %s positions" % entity)
        positions.append(after)
    result = np.asarray(positions, dtype=float)
    if result.ndim != 3 or result.shape[-1] != 2 or not np.isfinite(result).all():
        raise ValueError("Trajectory positions must be finite [H+1, nodes, 2]")
    return episode, result


def render_trajectories(run_dir: Path, checkpoint_kind="best"):
    root = Path(run_dir).resolve()
    directory = _evaluation_dir(root, checkpoint_kind)
    result = _json(directory/"summary.json", {})
    summary = result.get("summary", result) if isinstance(result, dict) else {}
    rows = _trajectory_rows(directory/"trajectories.jsonl")
    metadata = {"checkpoint_kind": checkpoint_kind, "figures": [], "warnings": [],
                "evaluation_dir": str(directory), "sources": []}
    if not rows:
        metadata["warnings"].append("No recorded %s evaluation trajectory." % checkpoint_kind)
        return metadata
    episode, uavs = extract_trajectory(rows, "uav")
    _, users = extract_trajectory(rows, "user")
    cfg = summary.get("environment_config", _config(root))
    area = _number(cfg.get("area_m"))
    if area is None or area <= 0:
        raise ValueError("Trajectory requires environment_config.area_m for physical map limits")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    for node in range(users.shape[1]):
        ax.plot(users[:, node, 0], users[:, node, 1], color="#AAAAAA", alpha=.5, linewidth=.65,
                label="User paths" if node == 0 else None, zorder=1)
    ax.scatter(users[0, :, 0], users[0, :, 1], marker=".", color="#777777", s=13,
               label="Users at t=0", zorder=2)
    for node in range(uavs.shape[1]):
        color = COLORS[node % len(COLORS)]
        x, y = uavs[:, node, 0], uavs[:, node, 1]
        ax.plot(x, y, color=color, linewidth=1.6, label="UAV %d" % (node+1), zorder=3)
        ax.scatter(x[0], y[0], facecolors="white", edgecolors=color, marker="o", s=65, linewidths=1.2, zorder=4, clip_on=False)
        ax.scatter(x[-1], y[-1], color=color, marker="X", s=35, zorder=4, clip_on=False)
        ax.annotate("q0-%d" % (node+1), (x[0], y[0]), xytext=(4, -11), textcoords="offset points", fontsize=8)
        ax.annotate("qH-%d" % (node+1), (x[-1], y[-1]), xytext=(4, 5), textcoords="offset points", fontsize=8)
        # Arrows use recorded segments and make travel direction unambiguous.
        for step in sorted(set((max(1, (len(x)-1)//3), max(1, 2*(len(x)-1)//3)))):
            if step < len(x) and (x[step] != x[step-1] or y[step] != y[step-1]):
                ax.annotate("", xy=(x[step], y[step]), xytext=(x[step-1], y[step-1]),
                            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3})
    ax.set(xlim=(0, area), ylim=(0, area), xlabel="x (m)", ylabel="y (m)",
           title="%s evaluation, episode %s: q0 to qH (%d frames)" %
           (checkpoint_kind.capitalize(), episode, uavs.shape[0]-1))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.), fontsize=8)
    name = "trajectory_"+("last" if checkpoint_kind == "final" else checkpoint_kind)
    metadata["figures"] = _relative(_save(fig, root/"plots"/name, plt), root)
    metadata.update(episode=episode, frames=int(uavs.shape[0]-1), points=int(uavs.shape[0]),
                    uavs=int(uavs.shape[1]), users=int(users.shape[1]),
                    initial_uav_position=uavs[0].tolist(), final_uav_position=uavs[-1].tolist())
    metadata["sources"] = [(directory/"trajectories.jsonl").relative_to(root).as_posix()]
    _write_json(root/"plots"/(name+"_metadata.json"), metadata)
    return metadata


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pairing(result):
    summary = result.get("summary", {})
    config = summary.get("environment_config")
    if not isinstance(config, dict) or not config:
        raise ValueError("Comparison requires each evaluation's environment_config")
    scenarios = result.get("scenarios") or result.get("episodes")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Comparison requires recorded scenario hashes and episode IDs")
    pairs = []
    for row in scenarios:
        if "episode" not in row or not row.get("scenario_hash"):
            raise ValueError("Comparison requires recorded scenario hashes and episode IDs")
        pairs.append({key: row[key] for key in ("episode", "scenario_hash", "frame_scenario_hashes") if key in row})
    if len({r["episode"] for r in pairs}) != len(pairs):
        raise ValueError("Duplicate evaluation episode IDs")
    pairs.sort(key=lambda row: row["episode"])
    return _canonical(config), _canonical(pairs)


def _manifest_groups(batch):
    manifest = _json(batch/"experiment_manifest.json")
    if manifest is None:
        raise FileNotFoundError("Comparison requires experiment_manifest.json")
    if isinstance(manifest, list):
        groups = manifest
    else:
        groups = manifest.get("groups", manifest.get("runs", manifest.get("experiments")))
    if isinstance(groups, dict):
        groups = [dict(value, id=key) for key, value in groups.items()]
    if not isinstance(groups, list) or not groups:
        raise ValueError("experiment_manifest.json must declare nonempty groups")
    return groups


def _contained(root, path):
    candidate = Path(path)
    candidate = candidate.resolve() if candidate.is_absolute() else (root/candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("Manifest run directory must remain inside the batch")
    return candidate


COMPARISON_METRICS = (
    "episode_return", "weighted_delay_mean", "weighted_delay_sum_per_frame", "mean_delay_s",
    "deadline_completion", "deadline_violation_rate", "deadline_violation_cost",
    "total_energy_j", "total_propulsion_energy_j", "total_compute_energy_j",
    "budget_feasible", "deadline_feasible", "fully_feasible", "feasible_episodes",
    "energy_feasible_episodes", "deadline_feasible_episodes", "collision_attempt_frames",
    "boundary_attempt_frames", "allocation_invalid_rate", "solver_feasible_frame_fraction",
    "fallback_rate", "total_decision_mean_s", "total_decision_p95_s",
)


def _aggregate(summary, episodes, metric):
    value = _number(summary.get(metric))
    if value is not None:
        return value
    values = [_number(r.get(metric)) for r in episodes]
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def _batch_trajectories(loaded, root, output, checkpoint_kind, plt):
    """Shared physical axes expose every recorded group's actual q0..qH."""
    paths, warnings = [], []
    for group, run, directory, result in loaded:
        rows = _trajectory_rows(directory/"trajectories.jsonl")
        label = str(group.get("label", group.get("name", group.get("id"))))
        if not rows:
            warnings.append("No recorded trajectory for "+label)
            continue
        episode, uavs = extract_trajectory(rows, "uav")
        _, users = extract_trajectory(rows, "user")
        area = _number(result["summary"]["environment_config"].get("area_m"))
        if area is None or area <= 0:
            raise ValueError("Trajectory requires environment_config.area_m")
        paths.append((label, episode, uavs, users, area))
    if not paths:
        return [], warnings
    columns = min(3, len(paths))
    nrows = (len(paths)+columns-1)//columns
    fig, axes = plt.subplots(nrows, columns, figsize=(4.0*columns, 3.75*nrows),
                             constrained_layout=True, squeeze=False)
    for ax, (label, episode, uavs, users, area) in zip(axes.flat, paths):
        for node in range(users.shape[1]):
            ax.plot(users[:, node, 0], users[:, node, 1], color="#AAAAAA", alpha=.45, linewidth=.5)
        ax.scatter(users[0, :, 0], users[0, :, 1], marker=".", s=8, color="#888888")
        for node in range(uavs.shape[1]):
            color = COLORS[node % len(COLORS)]
            x, y = uavs[:, node, 0], uavs[:, node, 1]
            ax.plot(x, y, color=color, linewidth=1.45)
            ax.scatter(x[0], y[0], marker="o", s=43, facecolors="white", edgecolors=color, linewidths=1.1, zorder=3, clip_on=False)
            ax.scatter(x[-1], y[-1], marker="X", s=25, color=color, zorder=4, clip_on=False)
            ax.annotate(str(node+1), (x[0], y[0]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set(xlim=(0, area), ylim=(0, area), xlabel="x (m)", ylabel="y (m)",
               title="%s\nEpisode %s" % (textwrap.fill(label, width=36), episode))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(paths):]:
        ax.set_visible(False)
    fig.suptitle("%s trajectories: all UAVs; circle = q0, cross = qH; gray = users" %
                 checkpoint_kind.capitalize(), fontsize=11)
    return _relative(_save(fig, output/"all_group_trajectories", plt), root), warnings


def _training_comparison_data(root, groups):
    """Check that the same completed-episode axis means the same frame budget."""
    histories, excluded, warnings = [], [], []
    reference_schedule, reference_config = None, None
    reference_fields = None
    schedule_fields = ("episode", "start_step", "end_step", "frames", "scenario_hash")
    for group in groups:
        identifier = str(group.get("id", group.get("name", group.get("label", ""))))
        if str(group.get("kind", "train")).lower() in ("baseline", "heuristic", "local"):
            excluded.append(identifier)
            continue
        run = _contained(root, group.get("run_dir", group.get("path", group.get("out", identifier))))
        rows = _training_rows(run)
        if not rows:
            warnings.append("No completed training episodes for "+identifier)
            continue
        available = tuple(field for field in schedule_fields if all(r.get(field) is not None for r in rows))
        if "episode" not in available:
            raise ValueError("Training comparison requires terminal episode IDs for "+identifier)
        if len({row["episode"] for row in rows}) != len(rows):
            raise ValueError("Duplicate terminal training episode IDs for "+identifier)
        if "end_step" in available:
            steps = [_number(row["end_step"]) for row in rows]
            if any(step is None for step in steps) or any(b <= a for a, b in zip(steps, steps[1:])):
                raise ValueError("Training end_step must strictly increase for "+identifier)
        schedule = [{field: row[field] for field in available} for row in rows]
        if reference_schedule is None:
            reference_schedule, reference_fields = schedule, available
        else:
            if available != reference_fields:
                raise ValueError("Cannot verify shared training episode/step axis: schedule columns differ for "+identifier)
            common = min(len(schedule), len(reference_schedule))
            if schedule[:common] != reference_schedule[:common]:
                raise ValueError("Training episode/step schedules or scenario hashes differ for "+identifier)
            if len(schedule) > len(reference_schedule):
                reference_schedule = schedule
        config = _config(run)
        if config:
            if reference_config is not None and _canonical(config) != reference_config:
                raise ValueError("Training environment configurations differ for "+identifier)
            reference_config = _canonical(config)
        histories.append({"id": identifier, "label": str(group.get("label", group.get("name", identifier))),
                          "control": group.get("control", "pure"), "rows": rows,
                          "source": (run/"episodes.csv").relative_to(root).as_posix()})
    return histories, {"excluded_baselines": excluded, "warnings": warnings,
                       "schedule_columns_checked": list(reference_fields or ()),
                       "shared_schedule_verified": bool(histories),
                       "axis": "Completed training episode, in completion order; identical recorded episode/step schedule over shared prefix"}


def render_training_comparison(batch_dir: Path, *, checkpoint_kind="best"):
    """Overlay actual learning histories; usable before any evaluation exists.

    Unequal lengths are permitted for a live batch, with no padding or invented
    tail. Shared prefixes must identify identical completed episodes and frame
    budgets. Baselines are explicitly excluded from the learning-curve figure.
    """
    if checkpoint_kind not in ("best", "last", "final"):
        raise ValueError("checkpoint_kind must be 'best' or 'last'")
    root = Path(batch_dir).resolve()
    histories, metadata = _training_comparison_data(root, _manifest_groups(root))
    metadata.update(figures=[], groups=len(histories), rolling_window=ROLLING_WINDOW,
                    sources=[history["source"] for history in histories],
                    training_seeds_per_group=1,
                    series=[{"group": history["id"], "label": history["label"],
                             "completed_episodes": len(history["rows"]),
                             "episode_ids": [row["episode"] for row in history["rows"]],
                             "end_steps": [row.get("end_step") for row in history["rows"]],
                             "x": list(range(1, len(history["rows"])+1))} for history in histories])
    if not histories:
        return metadata
    output = root/"comparison"/("last" if checkpoint_kind == "final" else checkpoint_kind)
    plt = _pyplot()
    # A separate legend strip keeps long method names away from data panels.
    fig = plt.figure(figsize=(12.0, 8.6), constrained_layout=True)
    layout = fig.add_gridspec(3, 3, height_ratios=(1., 1., .55))
    axes = [fig.add_subplot(layout[row, col]) for row in range(2) for col in range(3)]
    legend_ax = fig.add_subplot(layout[2, :])
    legend_ax.axis("off")
    panels = [("episode_return", "Episode return"), ("weighted_delay_mean", "J: normalized weighted delay"),
              ("total_energy_j", "Episode energy (J)"), ("deadline_completion", "Deadline completion fraction"),
              ("penalty_sum", "Total episode penalty"), ("fully_feasible", "Fully feasible episode (0/1)")]
    handles, labels = [], []
    for index, history in enumerate(histories):
        color = TRAINING_COLORS[index % len(TRAINING_COLORS)]
        style = "--" if history["control"] == "hybrid" else "-"
        x = list(range(1, len(history["rows"])+1))
        for ax, (metric, ylabel) in zip(axes, panels):
            values = [_penalty(row) if metric == "penalty_sum" else _number(row.get(metric))
                      for row in history["rows"]]
            numeric = [value if value is not None else float("nan") for value in values]
            ax.plot(x, numeric, color=color, alpha=.17, linewidth=.6, linestyle=style)
            line, = ax.plot(x, rolling_mean(numeric), color=color, linewidth=1.5, linestyle=style,
                            marker="o" if len(x) == 1 else None, markersize=3)
            if ax is axes[0]:
                handles.append(line)
                labels.append(textwrap.fill(history["label"], width=37))
            ax.set(xlabel="Completed training episode (completion order)", ylabel=ylabel)
            ax.grid(alpha=.18)
            if metric in ("deadline_completion", "fully_feasible"):
                ax.set_ylim(-.025, 1.025)
    legend_ax.legend(handles, labels, loc="center", ncol=3, fontsize=8,
                     handlelength=2.7, columnspacing=2.0, labelspacing=1.3)
    fig.suptitle("Training comparison: raw episodes and trailing 25-episode means\n"
                 "One training seed per method; baselines have no learning curves", fontsize=11)
    metadata["figures"] = _relative(_save(fig, output/"training_comparison", plt), root)
    _write_json(output/"training_comparison_metadata.json", metadata)
    return metadata


def render_comparison(batch_dir: Path, *, checkpoint_kind="best"):
    """Compare every declared group only after proving paired scenario inputs.

    Single training seeds have no error bars. Means across evaluation episodes
    are descriptive, and cannot stand in for independent training replications.
    """
    root = Path(batch_dir).resolve()
    groups = _manifest_groups(root)
    loaded = []
    reference = None
    for group in groups:
        identifier = str(group.get("id", group.get("name", group.get("label", ""))))
        if not identifier:
            raise ValueError("Every experiment group requires id or name")
        run = _contained(root, group.get("run_dir", group.get("path", group.get("out", identifier))))
        baseline = str(group.get("kind", "train")).lower() in ("baseline", "heuristic", "local")
        directory = _evaluation_dir(run, checkpoint_kind, baseline=baseline)
        result = _json(directory/"summary.json")
        if not isinstance(result, dict) or "summary" not in result:
            raise FileNotFoundError("Missing completed %s evaluation for %s: %s" %
                                    (checkpoint_kind, identifier, directory/"summary.json"))
        pairing = _pairing(result)
        if reference is None:
            reference = pairing
        elif pairing != reference:
            raise ValueError("Unpaired comparison: environment configuration or scenario hashes differ for "+identifier)
        loaded.append((group, run, directory, result))
    # Validate training axes before writing any evaluation comparison outputs.
    _training_comparison_data(root, groups)
    # No comparison table/figure is written until every declared group passes.
    rows = []
    for group, run, directory, result in loaded:
        summary, episodes = result["summary"], result.get("episodes", [])
        row = {"group": str(group.get("id", group.get("name", group.get("label")))),
               "label": group.get("label", group.get("name", group.get("id"))),
               "kind": group.get("kind", "train"), "control": group.get("control", summary.get("control")),
               "checkpoint_kind": "baseline" if str(group.get("kind", "")).lower() == "baseline" or summary.get("baseline") else checkpoint_kind,
               "evaluation_episodes": int(summary.get("episodes", len(episodes))),
               "training_seed": group.get("seed", summary.get("seed")), "training_seeds": 1,
               "source_summary": (directory/"summary.json").relative_to(root).as_posix(),
               "training_curve": "not applicable" if str(group.get("kind", "")).lower() == "baseline" or summary.get("baseline") else
                                 "available" if (run/"episodes.csv").is_file() else "not recorded"}
        for metric in COMPARISON_METRICS:
            row[metric] = _aggregate(summary, episodes, metric)
        vectors = [r.get("per_uav_energy_j") for r in episodes]
        n_uavs = max((len(v) for v in vectors if isinstance(v, list)), default=0)
        for node in range(n_uavs):
            values = [_number(v[node]) for v in vectors if isinstance(v, list) and len(v)>node]
            values = [v for v in values if v is not None]
            row["uav_%d_energy_j" % (node+1)] = sum(values)/len(values) if values else None
        rows.append(row)
    output = root/"comparison"/("last" if checkpoint_kind == "final" else checkpoint_kind)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = output/"summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output/"summary.csv")
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(max(10.5, .85*len(rows)), 7.2), constrained_layout=True)
    metrics = [("weighted_delay_mean", "J: normalized weighted delay"),
               ("episode_return", "Episode return"), ("total_energy_j", "Episode energy (J)"),
               ("deadline_completion", "Deadline completion fraction"),
               ("fully_feasible", "Fully feasible episode fraction"),
               ("total_decision_mean_s", "Decision time (s/frame)")]
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        positions, values, labels, colors = [], [], [], []
        for index, row in enumerate(rows):
            value = _number(row.get(metric))
            if value is None:
                ax.text(index, 0, "N/A", rotation=90, ha="center", fontsize=7)
                continue
            positions.append(index); values.append(value); labels.append(row["label"])
            colors.append(COLORS[0] if row["control"] == "hybrid" else COLORS[1]
                          if str(row["kind"]).lower() == "train" else COLORS[4])
        ax.bar(positions, values, color=colors, width=.65)
        ax.set_xticks(list(range(len(rows))))
        ax.set_xticklabels([str(r["label"]) for r in rows], rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.18)
        if metric in ("deadline_completion", "fully_feasible"):
            ax.set_ylim(0, 1.05)
    fig.suptitle("Paired %s evaluation; one training seed per group; no error bars" % checkpoint_kind, fontsize=11)
    figures = _relative(_save(fig, output/"all_groups", plt), root)
    trajectory_figures, trajectory_warnings = _batch_trajectories(loaded, root, output, checkpoint_kind, plt)
    figures += trajectory_figures
    training_metadata = render_training_comparison(root, checkpoint_kind=checkpoint_kind)
    figures += training_metadata["figures"]
    metadata = {"checkpoint_kind": checkpoint_kind, "paired": True, "groups": len(rows),
                "pairing_sha256": hashlib.sha256((reference[0]+reference[1]).encode()).hexdigest(),
                "statistics": "Means of paired evaluation episodes; one training seed per group; no error bars or significance claims.",
                "figures": figures, "table": (output/"summary.csv").relative_to(root).as_posix(),
                "rows": rows, "sources": [row["source_summary"] for row in rows],
                "warnings": trajectory_warnings+training_metadata["warnings"],
                "training_comparison": training_metadata}
    _write_json(output/"report_metadata.json", metadata)
    return metadata
