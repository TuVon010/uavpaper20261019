"""Strict readers for the archived seed17 plots; never modify raw results.

The three plotted metrics retain their recorded definitions: normalized weighted
delay (J), average task delay in seconds, and total energy of all three UAVs.
Only ordinary trailing means are provided for single-seed training curves.  A
smoothed line is not a multi-seed confidence interval.
"""
from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METRIC_COLUMNS = ("weighted_delay_mean", "mean_delay_s", "total_energy_j")


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_csv(path: Path) -> list[dict]:
    """Read CSV without coercing booleans, IDs, hashes, or numeric strings."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"Malformed CSV row: {path}")
        return rows


def num(value: Any) -> float:
    """Convert a scalar, including recorded boolean strings, and reject NaN/Inf."""
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        result = float(value.strip().lower() == "true")
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric: {value!r}")
    return result


def active_methods(config: dict, requested: list[str] | None = None) -> list[dict]:
    methods = config["methods"]
    ids = [method["id"] for method in methods]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate method IDs in plotting configuration")
    enabled = [method for method in methods if method.get("enabled", True) is True]
    if requested is None:
        return enabled
    if isinstance(requested, str) or len(requested) != len(set(requested)):
        raise ValueError("Requested methods must be a list of unique IDs")
    active = {method["id"] for method in enabled}
    invalid = set(requested) - active
    if invalid:
        raise ValueError(f"Unknown or disabled methods: {sorted(invalid)}")
    return [method for method in enabled if method["id"] in requested]


def view_methods(config: dict, view: str, requested: list[str] | None = None) -> list[dict]:
    methods = active_methods(config, requested)
    selected = config["views"][view].get("methods")
    if selected is None:
        return methods
    known = {method["id"] for method in config["methods"]}
    if isinstance(selected, str) or len(selected) != len(set(selected)) or set(selected) - known:
        raise ValueError(f"Invalid method IDs in view {view!r}: {selected!r}")
    return [method for method in methods if method["id"] in selected]


def run_dir(root: Path, case_id: str, method: str | dict) -> Path:
    root = Path(root).resolve()
    method_id = method["id"] if isinstance(method, dict) else method
    matches = [job for job in read_json(root / "suite_plan.json")["jobs"]
               if job["case_id"] == case_id and job["method"] == method_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one planned job: {case_id}/{method_id}")
    result = (root / matches[0]["run_dir"]).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Job directory escapes suite root: {result}")
    if not result.is_dir():
        raise FileNotFoundError(result)
    return result


def evaluation_dir(run: Path, checkpoint: str) -> Path:
    """Resolve the exact checkpoint; baselines alone share the evaluation alias."""
    if checkpoint not in {"best", "last"}:
        raise ValueError(f"Checkpoint must be best or last, got {checkpoint!r}")
    kind = read_json(Path(run) / "experiment.json")["kind"]
    if kind not in {"baseline", "train"}:
        raise ValueError(f"Unsupported experiment kind: {kind!r}")
    directory = Path(run) / ("evaluation" if kind == "baseline" else f"evaluation_{checkpoint}")
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return directory


def evaluation_records(run: Path, checkpoint: str) -> dict:
    directory = evaluation_dir(run, checkpoint)
    summary = read_json(directory / "summary.json")["summary"]
    episodes = read_csv(directory / "episodes.csv")
    if len(episodes) != 10 or int(num(summary["episodes"])) != 10:
        raise ValueError(f"Expected all ten archived evaluation episodes: {directory}")
    episode_ids = [num(row["episode"]) for row in episodes]
    if len(set(episode_ids)) != 10:
        raise ValueError(f"Duplicate evaluation episodes: {directory}")
    for column in METRIC_COLUMNS:
        actual = math.fsum(num(row[column]) for row in episodes) / len(episodes)
        expected = num(summary[column])
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"Episode mean disagrees with summary for {column}: {directory}")
    return {"summary": summary, "episodes": episodes, "directory": directory}


def training_records(run: Path) -> list[dict]:
    if read_json(Path(run) / "experiment.json")["kind"] == "baseline":
        return []
    rows = read_csv(Path(run) / "episodes.csv")
    if not rows:
        raise ValueError(f"No training episodes: {run}")
    return rows


def rolling_mean(values: Iterable[float], window: int) -> list[float]:
    """Trailing mean only; initial points use the available sample count."""
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("Rolling window must be a positive integer")
    queue: deque[float] = deque(maxlen=window)
    result = []
    for value in values:
        queue.append(num(value))
        result.append(math.fsum(queue) / len(queue))
    return result


def trajectory_frames(run: Path, checkpoint: str, episode: int) -> list[dict]:
    directory = evaluation_dir(run, checkpoint)
    frames = []
    with (directory / "trajectories.jsonl").open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if int(record["episode"]) == int(episode):
                    frames.append(record)
    frames.sort(key=lambda record: int(record["frame"]))
    frame_ids = [int(record["frame"]) for record in frames]
    if frame_ids != list(range(60)):
        raise ValueError(f"Expected exactly frames 0..59 for episode {episode}: {directory}")
    return frames


def trajectory_positions(frames: list[dict], entity: str = "uav") -> np.ndarray:
    """Return the true initial point and sixty recorded post-motion positions."""
    if entity not in {"uav", "user"}:
        raise ValueError(f"Entity must be uav or user: {entity!r}")
    if len(frames) != 60 or [int(row["frame"]) for row in frames] != list(range(60)):
        raise ValueError("Trajectory needs the complete ordered sixty-frame episode")
    pre_key, post_key, next_key = f"{entity}_position", f"{entity}_xy_m", f"next_{entity}_position"
    initial = np.asarray(frames[0][pre_key], dtype=float)
    if initial.ndim != 2 or initial.shape[1] != 2 or not np.isfinite(initial).all():
        raise ValueError(f"Invalid recorded initial {entity} position")
    positions = [initial]
    for frame in frames:
        if post_key not in frame and next_key not in frame:
            raise KeyError(f"Missing recorded post-motion {entity} position at frame {frame['frame']}")
        post = np.asarray(frame[post_key] if post_key in frame else frame[next_key], dtype=float)
        if post.shape != initial.shape or not np.isfinite(post).all():
            raise ValueError(f"Invalid {entity} position at frame {frame['frame']}")
        if post_key in frame and next_key in frame:
            alias = np.asarray(frame[next_key], dtype=float)
            if alias.shape != post.shape or not np.allclose(alias, post, rtol=0, atol=1e-9):
                raise ValueError(f"Post-motion {entity} position aliases disagree")
        if pre_key in frame:
            pre = np.asarray(frame[pre_key], dtype=float)
            if pre.shape != initial.shape or not np.allclose(pre, positions[-1], rtol=0, atol=1e-9):
                raise ValueError(f"Discontinuous recorded {entity} trajectory at frame {frame['frame']}")
        positions.append(post)
    return np.stack(positions)
