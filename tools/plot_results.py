"""Render archived results only: no simulator, torch, model selection or training.

python tools/plot_results.py --cases main
python tools/plot_results.py --methods hybrid_ppo hybrid_ppo_ar_adaptive --output-tag selected
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from plot_support import (active_methods, evaluation_records, num, read_csv,
    read_json, rolling_mean, run_dir, training_records, trajectory_frames,
    trajectory_positions, view_methods)

METRICS = [("weighted_delay_mean", "Mean weighted delay cost", 1.),
           ("mean_delay_s", "Mean task delay (s)", 1.),
           ("total_energy_j", "Total UAV energy (kJ)", .001)]
AXES = {"users": "Number of user devices",
        "mean_data_mbit": "Mean task data size (Mbit)",
        "uav_cpu_ghz": "CPU capacity per UAV (GHz)",
        "uav_bandwidth_mhz": "Bandwidth per UAV (MHz)"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.color": "#d3d7dc", "grid.alpha": .65,
        "grid.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "lines.linewidth": 1.6})


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(fig, base, rows, root, config, title, note, kind, **extra):
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in config["formats"]:
        path = base.with_suffix("." + ext)
        fig.savefig(path, dpi=config["dpi"], bbox_inches="tight", pad_inches=.14)
        paths.append(path.relative_to(root).as_posix())
    plt.close(fig)
    source = base.parent / "source_data" / (base.name + ".csv")
    csv_write(source, rows)
    return dict(title=title, kind=kind, note=note, files=paths,
                source_csv=source.relative_to(root).as_posix(), **extra)


def legend(fig, methods, bars=False, bottom=.02):
    handles = [Patch(facecolor=m["color"], edgecolor="#333333", hatch=m["hatch"],
                     label=m["label"]) if bars else
               Line2D([], [], color=m["color"], marker=m["marker"],
                      linestyle=m["linestyle"], label=m["label"], markersize=5)
               for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, bottom),
               ncol=min(4, len(methods)), frameon=False)


def train_figure(root, out, case_id, methods, runs, config, view, symlog=False):
    methods = [m for m in methods if runs[m["id"]]["training"]]
    if not methods:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    rows = []
    for m in methods:
        training = runs[m["id"]]["training"]
        y = [num(r["episode_return"]) for r in training]
        smooth = rolling_mean(y, config["training"]["smoothing_window"])
        x = np.arange(1, len(y) + 1)
        if config["training"]["show_raw"]:
            ax.plot(x, y, color=m["color"], alpha=config["training"]["raw_alpha"], lw=.65)
        ax.plot(x, smooth, color=m["color"], linestyle=m["linestyle"], lw=1.6)
        for i, (raw, mean, record) in enumerate(zip(y, smooth, training), 1):
            rows.append(dict(method=m["id"], completed_episode=i,
                original_episode_id=record["episode"], end_step=record["end_step"],
                raw_return=raw, trailing_mean=mean,
                smoothing_window=config["training"]["smoothing_window"]))
    ax.set(xlabel="Completed training episodes", ylabel="Episode return (higher is better)",
           title=f"{case_id} | {config['views'][view]['label']}")
    if symlog:
        ax.set_yscale("symlog", linthresh=100)
        ax.set_ylabel("Episode return (symmetric log scale)")
    ax.margins(x=.005)
    legend(fig, methods)
    fig.subplots_adjust(bottom=.24, top=.9)
    note = (f"One training seed ({config['_protocol']['training_seed']}); faint = raw episode return; bold = trailing "
            f"mean of {config['training']['smoothing_window']} completed episodes. "
            "No clipping, omitted outliers, or multi-seed confidence intervals.")
    name = f"training_reward_{view}" + ("_symlog" if symlog else "")
    return save(fig, out / name, rows, root, config, name, note, "training", case=case_id)


def bar_figure(root, out, case_id, methods, runs, config, view, checkpoint, selected=False):
    if not methods:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8))
    source = []
    for ax, (metric, ylabel, scale) in zip(axes, METRICS):
        for i, m in enumerate(methods):
            records = runs[m["id"]][checkpoint]["episodes"]
            if selected:
                records = [r for r in records if int(num(r["episode"])) == config["trajectories"]["episode"]]
                if len(records) != 1:
                    raise ValueError("Selected evaluation episode is missing")
            values = np.asarray([num(r[metric]) for r in records]) * scale
            mean = float(np.mean(values))
            error = None
            if not selected and config["performance"]["error_bars"] == "range":
                error = np.asarray([[mean - values.min()], [values.max() - mean]])
            ax.bar(i, mean, width=.68, color=m["color"], edgecolor="#333333", linewidth=.6,
                   hatch=m["hatch"], yerr=error, capsize=3,
                   error_kw={"elinewidth": 1., "capthick": 1., "ecolor": "#444444"})
            feasible = sum(num(r["budget_feasible"]) > .5 for r in records)
            if metric == "total_energy_j" and config["performance"]["annotate_feasibility"]:
                ax.annotate(f"{feasible}/{len(records)}", (i, values.max()),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
            for record in records:
                source.append(dict(case_id=case_id, method=m["id"], checkpoint=checkpoint,
                    episode=record["episode"], metric=metric, recorded_value=record[metric],
                    displayed_value=num(record[metric]) * scale,
                    budget_feasible=record["budget_feasible"]))
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(methods)), [m["short_label"] for m in methods], rotation=35, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        ax.margins(y=.18)
        if metric == "total_energy_j":
            budget = runs[methods[0]["id"]]["environment"]["energy_budget_j"] * 3 / 1000
            ax.axhline(budget, color="#555555", lw=.8, ls=":")
    population = f"single episode {config['trajectories']['episode']}" if selected else "10 paired evaluation episodes"
    fig.suptitle(f"{case_id} | {checkpoint} | {population} | {config['views'][view]['label']}", fontsize=11)
    legend(fig, methods, bars=True, bottom=-.005)
    fig.subplots_adjust(bottom=.29, top=.86, wspace=.3)
    note = ("Bars = recorded scenario means; whiskers = scenario min-max (not confidence intervals). "
            if not selected else "Bars show one selected scenario only; no uncertainty estimate. ")
    note += ("Energy labels = energy-feasible episodes / total, checked separately per UAV. "
             "Dotted total-budget line is only a necessary condition; below it does not guarantee feasibility.")
    name = f"metrics_{checkpoint}_{view}" + ("_selected_scene" if selected else "")
    return save(fig, out / name, source, root, config, name, note, "performance",
                case=case_id, checkpoint=checkpoint)


def draw_trajectory(ax, frames, config, environment, title):
    users = trajectory_positions(frames, "user")
    uavs = trajectory_positions(frames, "uav")
    for k in range(users.shape[1]):
        ax.plot(users[:, k, 0], users[:, k, 1], color="#888888", lw=.45, alpha=.4)
    ax.scatter(users[0, :, 0], users[0, :, 1], color="#777777", s=8, alpha=.55)
    for k, color in enumerate(config["trajectories"]["uav_colors"]):
        q = uavs[:, k]
        ax.plot(q[:, 0], q[:, 1], color=color, lw=1.5)
        ax.scatter(*q[0], color="white", edgecolors=color, linewidths=1.3, s=36, zorder=4)
        ax.scatter(*q[-1], color=color, marker="*", s=65, zorder=5)
    size = environment["area_m"]
    margin = size * .02
    ax.set(xlim=(-margin, size + margin), ylim=(-margin, size + margin),
           xticks=np.linspace(0, size, 6), yticks=np.linspace(0, size, 6),
           xlabel="x (m)", ylabel="y (m)", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)


def trajectory_source(method, frames):
    rows = []
    for entity in ("uav", "user"):
        for t, positions in enumerate(trajectory_positions(frames, entity)):
            for k, p in enumerate(positions):
                rows.append(dict(method=method, time_s=t, entity=entity, entity_id=k,
                                 x_m=p[0], y_m=p[1]))
    return rows


def trajectory_figure(root, out, case_id, methods, runs, config, checkpoint):
    cols = min(4, len(methods))
    nrows = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.7 * cols, 3.8 * nrows), squeeze=False)
    source = []
    for ax, m in zip(axes.flat, methods):
        frames = runs[m["id"]]["frames"][checkpoint]
        draw_trajectory(ax, frames, config, runs[m["id"]]["environment"], m["label"])
        source.extend(trajectory_source(m["id"], frames))
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=c, label=f"UAV {k+1}") for k, c in enumerate(config["trajectories"]["uav_colors"])]
    handles += [Line2D([], [], color="#888888", label="Users"),
                Line2D([], [], marker="o", markerfacecolor="white", color="#555555", ls="", label="Start"),
                Line2D([], [], marker="*", color="#555555", ls="", label="End")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(f"{case_id} | {checkpoint} | episode {config['trajectories']['episode']}")
    fig.subplots_adjust(bottom=.09 if nrows > 1 else .17, top=.91, hspace=.34, wspace=.3)
    name = f"trajectories_{checkpoint}"
    return save(fig, out / name, source, root, config, name,
                "Actual initial point plus 60 post-motion points; slot-start service, motion affects next slot.",
                "trajectory", case=case_id, checkpoint=checkpoint)


def overview(root, case_id, m, data, config, tag):
    checkpoint = config["per_run"]["checkpoint"]
    frames = data["frames"][checkpoint]
    fig = plt.figure(figsize=(12.5, 7.4))
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[0, 0])
    training = data["training"]
    rows = []
    if training:
        values = [num(r["episode_return"]) for r in training]
        x = np.arange(1, len(values) + 1)
        smooth = rolling_mean(values, config["training"]["smoothing_window"])
        ax.plot(x, values, color=m["color"], alpha=.17, lw=.65)
        ax.plot(x, smooth, color=m["color"])
        ax.set(xlabel="Completed training episodes", ylabel="Episode return", title="Training (raw + trailing mean)")
        for i, (v, s) in enumerate(zip(values, smooth), 1):
            rows.append(dict(record_type="training", index=i, raw_return=v, trailing_mean=s))
    else:
        ax.plot(range(60), [r["reward"] for r in frames], color=m["color"])
        ax.set(xlabel="Slot", ylabel="Slot reward", title="Baseline evaluation (no training)")
    for col, key, label in [(1, "normalized_weighted_delay", "Weighted delay cost"),
                            (2, "mean_delay_s", "Mean task delay (s)")]:
        axis = fig.add_subplot(grid[0, col])
        axis.plot(range(60), [r[key] for r in frames], color=m["color"])
        axis.set(xlabel="Slot", ylabel=label, title=f"Episode {config['trajectories']['episode']}")
    energy = fig.add_subplot(grid[1, 0])
    cumulative = np.asarray([r["cumulative_energy_j"] for r in frames]) / 1000
    for k, color in enumerate(config["trajectories"]["uav_colors"]):
        energy.plot(range(1, 61), cumulative[:, k], color=color, label=f"UAV {k+1}")
    energy.axhline(data["environment"]["energy_budget_j"] / 1000, color="#555555", ls=":", label="Per-UAV budget")
    energy.set(xlabel="Time (s)", ylabel="Cumulative energy (kJ)")
    energy.legend(fontsize=8, frameon=False)
    traj = fig.add_subplot(grid[1, 1:])
    draw_trajectory(traj, frames, config, data["environment"], "UAV and user trajectories (start o / end star)")
    summary = data[checkpoint]["summary"]
    fig.suptitle(f"{case_id} | {m['label']} | {checkpoint}\n"
                 f"10-episode means: J={num(summary['weighted_delay_mean']):.4g}, "
                 f"delay={num(summary['mean_delay_s']):.4g} s, energy={num(summary['total_energy_j']) / 1000:.4g} kJ", fontsize=11)
    fig.subplots_adjust(top=.84, bottom=.08, hspace=.42, wspace=.36)
    for r in frames:
        rows.append(dict(record_type="evaluation_frame", index=r["frame"], reward=r["reward"],
            weighted_delay_cost=r["normalized_weighted_delay"], mean_delay_s=r["mean_delay_s"],
            cumulative_energy_uav1_j=r["cumulative_energy_j"][0],
            cumulative_energy_uav2_j=r["cumulative_energy_j"][1],
            cumulative_energy_uav3_j=r["cumulative_energy_j"][2]))
    rows += [dict(record_type="trajectory", **r) for r in trajectory_source(m["id"], frames)]
    out = data["run"] / "figures" / tag
    return save(fig, out / f"overview_{checkpoint}", rows, root, config, m["label"] + " overview",
                "Top reward is training for learners, evaluation slots for baselines. Single-scene trajectory; header means use all ten evaluation episodes.",
                "per_run", case=case_id, method=m["id"], checkpoint=checkpoint)


def case_worker(payload):
    root_string, case_id, config, requested, groups, tag = payload
    root = Path(root_string)
    style()
    methods = active_methods(config, requested)
    out = root / "cases" / case_id / "figures" / tag
    runs = {}
    need_frames = "trajectories" in groups or ("per_run" in groups and config["per_run"]["enabled"])
    for m in methods:
        run = run_dir(root, case_id, m)
        data = {"run": run, "environment": read_json(run / "effective_environment.json"),
                "training": training_records(run), "frames": {}}
        for cp in {"best", "last"}:
            data[cp] = evaluation_records(run, cp)
        if need_frames:
            frame_checkpoints = set(config["trajectories"]["checkpoints"]) if "trajectories" in groups else set()
            if "per_run" in groups and config["per_run"]["enabled"]:
                frame_checkpoints.add(config["per_run"]["checkpoint"])
            for cp in frame_checkpoints:
                data["frames"][cp] = trajectory_frames(run, cp, config["trajectories"]["episode"])
        runs[m["id"]] = data
    figures = []
    for view in config["views"]:
        chosen = view_methods(config, view, requested)
        if not chosen:
            continue
        if "training" in groups:
            figures.append(train_figure(root, out, case_id, chosen, runs, config, view))
            if view == "all" and config["training"]["all_symlog_companion"]:
                figures.append(train_figure(root, out, case_id, chosen, runs, config, view, True))
        if "performance" in groups:
            for cp in config["performance"]["checkpoints"]:
                figures.append(bar_figure(root, out, case_id, chosen, runs, config, view, cp))
                if config["performance"]["selected_scene"]:
                    figures.append(bar_figure(root, out, case_id, chosen, runs, config, view, cp, True))
    if "trajectories" in groups:
        for cp in config["trajectories"]["checkpoints"]:
            figures.append(trajectory_figure(root, out, case_id, methods, runs, config, cp))
    if "per_run" in groups and config["per_run"]["enabled"]:
        figures.extend(overview(root, case_id, m, runs[m["id"]], config, tag) for m in methods)
    return [f for f in figures if f]


def sweep_figures(root, config, requested, tag):
    figures = []
    out = root / "figures" / tag
    for cp in config["sweeps"]["checkpoints"]:
        rows = read_csv(root / "reports" / f"sweep_{cp}.csv")
        for view in config["views"]:
            methods = view_methods(config, view, requested)
            if not methods:
                continue
            for axis_id, xlabel in AXES.items():
                fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
                source = []
                for m in methods:
                    points = sorted([r for r in rows if r["axis"] == axis_id and r["method"] == m["id"]],
                                    key=lambda r: num(r["value"]))
                    if len(points) != 5:
                        raise ValueError(f"Incomplete sensitivity curve: {axis_id}/{m['id']}")
                    x = [num(r["value"]) for r in points]
                    for ax, (metric, ylabel, scale) in zip(axes, METRICS):
                        ax.plot(x, [num(r[metric]) * scale for r in points], color=m["color"],
                                linestyle=m["linestyle"], marker=m["marker"], markersize=5)
                        ax.set(xlabel=xlabel, ylabel=ylabel, xticks=x)
                    source.extend(points)
                for ax in axes:
                    ax.relim()
                    ax.autoscale(enable=True, axis="y")
                    ax.set_ylim(bottom=0)
                fig.suptitle(f"{xlabel} | {cp} | 10 paired evaluation episodes | {config['views'][view]['label']}", fontsize=11)
                legend(fig, methods, bottom=.005)
                fig.subplots_adjust(bottom=.25, top=.86, wspace=.32)
                figures.append(save(fig, out / f"sweep_{axis_id}_{cp}_{view}", source, root, config,
                    f"sweep_{axis_id}_{cp}_{view}",
                    "Recorded ten-scenario means, straight segments only. One factor changes; main reused for users20/CPU10/BW10. No retraining or model reselection.",
                    "sweep", axis=axis_id, checkpoint=cp))
    return figures


def write_index(root, out, figures, config, cases, requested):
    files = {}
    for f in figures:
        for relative in f["files"] + [f["source_csv"]]:
            path = root / relative
            files[relative] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    manifest = dict(schema_version=1, path_root="suite", created_utc=datetime.now(timezone.utc).isoformat(),
        suite=root.name, **config["_protocol"],
        cases=cases, methods=[m["id"] for m in active_methods(config, requested)],
        config=config, figure_count=len(figures), figures=figures, files=files)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    sections = ["<html><head><meta charset='utf-8'><title>Archived experiment figures</title>",
        "<style>body{font-family:Arial;margin:30px;max-width:1500px}img{max-width:100%;height:auto}article{border-top:1px solid #ddd;padding:16px 0}nav a{margin:8px}small{color:#555}</style></head><body>",
        f"<h1>{html.escape(root.name)}: experiment figures</h1><p>One training seed ({config['_protocol']['training_seed']}); ten paired evaluation scenarios. Best and last are separate. Faint training lines are raw data; bold lines are trailing means. All source data and models are retained.</p>"]
    sections.append("<nav>" + " | ".join(f"<a href='#{html.escape(g)}'>{html.escape(g)}</a>" for g in ["sweeps"] + cases) + "</nav>")
    def link(relative):
        return os.path.relpath(root / relative, out).replace(os.sep, "/")
    for group in ["sweeps"] + cases:
        selected = [f for f in figures if (f["kind"] == "sweep" if group == "sweeps" else f.get("case") == group)]
        if not selected:
            continue
        sections.append(f"<h2 id='{html.escape(group)}'>{html.escape(group)}</h2>")
        for f in selected:
            links = " | ".join(f"<a href='{html.escape(link(p))}'>{Path(p).suffix[1:].upper()}</a>" for p in f["files"])
            links += f" | <a href='{html.escape(link(f['source_csv']))}'>Source CSV</a>"
            png = next((p for p in f["files"] if p.endswith(".png")), None)
            sections.append(f"<article><h3>{html.escape(f['title'])}</h3><p>{links}</p><small>{html.escape(f['note'])}</small>" +
                            (f"<p><img loading='lazy' src='{html.escape(link(png))}'></p>" if png else "") + "</article>")
    sections.append("</body></html>")
    (out / "index.html").write_text("\n".join(sections), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/figures.json"))
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--cases", nargs="+", help="Case IDs; default all")
    parser.add_argument("--methods", nargs="+", help="Enabled method IDs; default config order")
    parser.add_argument("--groups", nargs="+", choices=["training", "performance", "trajectories", "per_run", "sweeps"],
                        default=["training", "performance", "trajectories", "per_run", "sweeps"])
    parser.add_argument("--output-tag", help="Fresh figure variant name; default config output_tag")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--checkpoint", choices=["best", "last", "both"], default="both")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    config_path = args.config if args.config.is_absolute() else project / args.config
    config = read_json(config_path)
    root = args.suite or Path(config["suite"])
    root = (root if root.is_absolute() else project / root).resolve()
    plan = read_json(root / "suite_plan.json")
    protocol = plan["suite"]["evaluation"]
    config["_protocol"] = dict(training_seed=plan["suite"]["seed"],
        evaluation_episodes=list(range(protocol["episode_start"], protocol["episode_start"] + protocol["episodes"])))
    all_cases = [r["id"] for r in plan["cases"]]
    cases = args.cases or all_cases
    if len(cases) != len(set(cases)) or set(cases) - set(all_cases):
        parser.error("Unknown or duplicate case ID")
    if not active_methods(config, args.methods):
        parser.error("At least one method must be enabled")
    tag = args.output_tag or config["output_tag"]
    if not tag or Path(tag).name != tag or tag in {".", ".."}:
        parser.error("Output tag must be one directory name")
    if set(config["formats"]) - {"png", "svg", "pdf"}:
        parser.error("Supported formats: png svg pdf")
    if args.checkpoint != "both":
        for key in ("performance", "sweeps", "trajectories"):
            config[key]["checkpoints"] = [args.checkpoint]
        config["per_run"]["checkpoint"] = args.checkpoint
    style()
    payloads = [(str(root), case, config, args.methods, args.groups, tag) for case in cases]
    figures = []
    workers = args.workers or config.get("workers", 1)
    if workers < 1:
        parser.error("Workers must be positive")
    if set(args.groups) - {"sweeps"}:
        if workers == 1:
            for payload in payloads:
                figures.extend(case_worker(payload))
                print("Plotted " + payload[1], flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(case_worker, p): p[1] for p in payloads}
                for future in as_completed(futures):
                    figures.extend(future.result())
                    print("Plotted " + futures[future], flush=True)
    if "sweeps" in args.groups:
        figures.extend(sweep_figures(root, config, args.methods, tag))
    figures.sort(key=lambda f: (f.get("case", ""), f["kind"], f["title"]))
    out = root / "figures" / tag
    manifest = write_index(root, out, figures, config, cases, args.methods)
    print(json.dumps(dict(figures=manifest["figure_count"], files=len(manifest["files"]), index=str(out / "index.html"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
