"""Read-only verification fixtures; no original data or training is changed."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/verify_plot_release.py"
spec = importlib.util.spec_from_file_location("verify_plot_release", MODULE_PATH)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def describe(path):
    return dict(sha256=verify.sha_file(path), bytes=path.stat().st_size)


def exercise():
    checks = []
    with tempfile.TemporaryDirectory(prefix="vf") as directory:
        root = Path(directory)
        suite = root / "results/demo"
        fig = suite / "figures/paper/a.svg"
        fig.parent.mkdir(parents=True)
        fig.write_text("<svg/>", encoding="utf-8")
        csv = suite / "cases/raw.csv"
        csv.parent.mkdir(parents=True)
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        save(root / "CURRENT_SUITE.json", dict(result_directory="results/demo"))
        save(suite / "figures/paper/manifest.json", dict(path_root="suite", files={
            "figures/paper/a.svg": describe(fig)}))
        release = {str(p.relative_to(root).as_posix()): describe(p) for p in [root / "CURRENT_SUITE.json", fig, suite / "figures/paper/manifest.json"]}
        save(root / "MANIFEST.json", dict(files=release))
        save(root / "evidence/seed17_final_archive_manifest.json", dict(suite_id="demo", cases=[dict(case_id="main", files={
            "cases/raw.csv": describe(csv)})], metadata=dict(files={"collection.lock": dict(sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", bytes=0)}), result_bytes=csv.stat().st_size))
        (suite / "collection.lock").write_bytes(b"")
        before = {str(p.relative_to(root)): describe(p) for p in root.rglob("*") if p.is_file()}
        with redirect_stdout(io.StringIO()):
            assert verify.main(["--project", str(root)]) == 0
            assert verify.main(["--project", str(root), "--full-data"]) == 0
        after = {str(p.relative_to(root)): describe(p) for p in root.rglob("*") if p.is_file()}
        assert before == after
        checks.append("managed + suite-relative figure and raw checks pass without mutations")
        csv.write_text("altered", encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            assert verify.main(["--project", str(root)]) == 0
            assert verify.main(["--project", str(root), "--full-data"]) == 1
        checks.append("--full-data detects raw changes outside the quick managed scope")
        fig.write_text("bad", encoding="utf-8")
        capture = io.StringIO()
        with redirect_stdout(capture):
            assert verify.main(["--project", str(root)]) == 1
        assert json.loads(capture.getvalue())["passed"] is False
        checks.append("changed figure fails verification")
        for invalid in ["../escape", "/absolute", "C:/absolute", ".", "foo\\bar"]:
            try:
                verify.safe_file(root, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsafe path accepted: " + invalid)
        checks.append("unsafe manifest paths are rejected")
    return dict(passed=True, checks=checks)


def test_verifier_fixtures():
    assert exercise()["passed"]


if __name__ == "__main__":
    print(json.dumps(exercise(), indent=2))
