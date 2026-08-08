from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = PROJECT_ROOT / "models" / "hf_watch_top20_model.joblib"

PACKAGE_NAMES = [
    "joblib",
    "matplotlib",
    "neurokit2",
    "numpy",
    "pandas",
    "scikit-learn",
    "shap",
    "streamlit",
    "tqdm",
    "wfdb",
    "xgboost",
]

ARTIFACTS_TO_HASH = [
    MODEL_PATH,
    PROJECT_ROOT / "features" / "training_dataset_clean.csv",
    PROJECT_ROOT / "features" / "external_rr_features.csv",
    PROJECT_ROOT / "results" / "external_validation_predictions.csv",
    PROJECT_ROOT / "results" / "external_validation_record_summary.csv",
    PROJECT_ROOT / "results" / "nested_feature_selection_summary.csv",
    PROJECT_ROOT / "results" / "external_missingness_feature_summary.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return versions


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    commit = git_output("rev-parse", "HEAD")
    branch = git_output("rev-parse", "--abbrev-ref", "HEAD")
    dirty_lines = git_output("status", "--porcelain")
    dirty = bool(dirty_lines)

    versions = package_versions()

    artifact_rows: list[dict[str, object]] = []
    for path in ARTIFACTS_TO_HASH:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if path.exists():
            artifact_rows.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        else:
            artifact_rows.append(
                {
                    "path": relative,
                    "missing": True,
                }
            )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project": "HF-Watch-AI",
        "git": {
            "branch": branch,
            "commit": commit,
            "working_tree_dirty": dirty,
            "dirty_entries": dirty_lines.splitlines() if dirty_lines else [],
        },
        "runtime": {
            "python_version": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": versions,
        "artifacts": artifact_rows,
    }

    json_path = RESULTS_DIR / "reproducibility_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    freeze_path = RESULTS_DIR / "environment_freeze.txt"
    freeze_lines = [
        f"Python=={platform.python_version()}",
        *[f"{name}=={version}" for name, version in versions.items()],
    ]
    freeze_path.write_text("\n".join(freeze_lines) + "\n", encoding="utf-8")

    print("HF-Watch-AI reproducibility manifest")
    print("------------------------------------")
    print(f"Git branch: {branch}")
    print(f"Git commit: {commit}")
    print(f"Working tree dirty: {dirty}")
    if dirty and dirty_lines:
        print("Uncommitted/untracked entries:")
        for line in dirty_lines.splitlines():
            print(" ", line)

    print("\nRuntime:")
    print(" Python:", platform.python_version())
    print(" Platform:", platform.platform())
    print(" Machine:", platform.machine())
    print(" Processor:", platform.processor() or "[not reported by platform]")

    print("\nPackage versions:")
    for name, version in versions.items():
        print(f" {name}: {version}")

    print("\nArtifact SHA-256 checksums:")
    for item in artifact_rows:
        if item.get("missing"):
            print(f" MISSING: {item['path']}")
        else:
            print(f" {item['path']}: {item['sha256']}")

    print("\nSaved:")
    print(" -", json_path)
    print(" -", freeze_path)
    print("\nThis script does not modify the model, data, predictions, or thresholds.")


if __name__ == "__main__":
    main()
