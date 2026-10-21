"""Read-only verification of the plotting release and optional original data.

MANIFEST.json: {"files": {"project/relative/path": {"sha256": ..., "bytes": ...}}}
Default checks all managed release files and the plotting manifest when present.
--full-data additionally streams the original archive manifest's raw data files.
No plotting, training, report regeneration or metadata mutation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Manifest paths must be nonempty relative POSIX paths: " + repr(name))
    rel = PurePosixPath(name)
    if not rel.parts or rel.is_absolute() or ".." in rel.parts or ":" in rel.parts[0]:
        raise ValueError("Unsafe manifest path: " + name)
    root = Path(root).resolve()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Manifest path escapes selected root: " + name)
    return candidate


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root, entries, name, errors):
    if not isinstance(entries, dict) or not entries:
        errors.append(name + ": missing or empty files dictionary")
        return dict(files=0, bytes=0, matched=0)
    count, size, matched = 0, 0, 0
    for filename, expected in entries.items():
        count += 1
        try:
            if not isinstance(expected, dict) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected.get("sha256", "")):
                raise ValueError("Invalid SHA256 metadata")
            wanted_size = expected.get("bytes")
            if isinstance(wanted_size, bool) or not isinstance(wanted_size, int) or wanted_size < 0:
                raise ValueError("Invalid bytes metadata")
            path = safe_file(root, filename)
            if not path.is_file():
                raise FileNotFoundError("Missing file")
            actual_size = path.stat().st_size
            size += actual_size
            if actual_size != wanted_size:
                raise ValueError(f"Size differs: expected {wanted_size}, actual {actual_size}")
            if sha_file(path) != expected["sha256"].lower():
                raise ValueError("SHA256 differs")
            matched += 1
        except (ValueError, OSError, TypeError) as error:
            errors.append(name + "/" + str(filename) + ": " + str(error))
    return dict(files=count, bytes=size, matched=matched)


def figure_entries(payload):
    """Support an ordinary files dictionary or an explicitly rooted flat list."""
    if isinstance(payload.get("files"), dict):
        return payload["files"]
    figures = payload.get("figures")
    if not isinstance(figures, list) or not figures:
        raise ValueError("Plot manifest needs a files dictionary or nonempty figures list")
    entries = {}
    for item in figures:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Figure entries need path, sha256 and bytes")
        if item["path"] in entries:
            raise ValueError("Duplicate figure path: " + item["path"])
        entries[item["path"]] = dict(sha256=item.get("sha256"), bytes=item.get("bytes"))
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default="MANIFEST.json")
    parser.add_argument("--figures-manifest", help="Project-relative plotting manifest; autodetected from CURRENT_SUITE.json")
    parser.add_argument("--full-data", action="store_true", help="Additionally hash original case + suite metadata files (~17.1 GB)")
    parser.add_argument("--raw-manifest", default="evidence/seed17_final_archive_manifest.json")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    errors, checks = [], {}
    try:
        manifest = read_json(safe_file(project, args.manifest))
        checks["release"] = verify_files(project, manifest.get("files"), "release", errors)
        # Release manifests can identify the plotting manifest explicitly. The
        # default path follows the selected suite, never an arbitrary glob.
        pointer = read_json(project / "CURRENT_SUITE.json") if (project / "CURRENT_SUITE.json").is_file() else {}
        suite_name = manifest.get("suite_directory") or pointer.get("result_directory")
        figure_name = args.figures_manifest or manifest.get("figures_manifest")
        if figure_name is None and suite_name:
            figure_name = str(PurePosixPath(suite_name) / "figures/paper/manifest.json")
        if figure_name:
            figure_path = safe_file(project, figure_name)
            payload = read_json(figure_path)
            path_root = payload.get("path_root", "project")
            roots = {"project": project, "manifest": figure_path.parent}
            if suite_name:
                roots["suite"] = safe_file(project, suite_name)
            if path_root not in roots:
                raise ValueError("Unknown figure manifest path_root: " + str(path_root))
            checks["figures"] = verify_files(roots[path_root], figure_entries(payload), "figures", errors)
        else:
            raise ValueError("Cannot locate plotting manifest; provide --figures-manifest")
        if args.full_data:
            raw = read_json(safe_file(project, args.raw_manifest))
            suite = safe_file(project, "results/" + raw["suite_id"])
            combined = {}
            for case in raw["cases"]:
                for filename, metadata in case["files"].items():
                    if filename in combined:
                        raise ValueError("Duplicate raw manifest path: " + filename)
                    combined[filename] = metadata
            for filename, metadata in raw["metadata"]["files"].items():
                if filename in combined:
                    raise ValueError("Duplicate raw metadata path: " + filename)
                combined[filename] = metadata
            checks["raw_data"] = verify_files(suite, combined, "raw_data", errors)
            if checks["raw_data"]["bytes"] != raw.get("result_bytes"):
                errors.append("raw_data: byte total differs from archive manifest")
            checks["raw_data"].update(cases=len(raw["cases"]), suite_id=raw["suite_id"])
    except (ValueError, OSError, KeyError, TypeError) as error:
        errors.append(str(error))
    result = dict(passed=not errors, project=str(project), full_data=args.full_data, checks=checks, errors=errors,
        policy="Read-only SHA256/size verification; no training or metadata changes.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
