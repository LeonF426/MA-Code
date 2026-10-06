"""Reproducible, non-overwriting artifact directories for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=lambda item: str(item),
    ) + "\n"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "run"


def _git_output(arguments: list[str], cwd: Path, *, binary: bool = False):
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    candidates = [root / "pyproject.toml"]
    for directory in (root / "src", root / "examples"):
        if directory.is_dir():
            candidates.extend(sorted(directory.rglob("*.py")))
    for path in sorted({item.resolve() for item in candidates if item.is_file()}):
        digest.update(path.relative_to(root.resolve()).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _provenance(cwd: Path) -> dict[str, Any]:
    root_text = _git_output(["rev-parse", "--show-toplevel"], cwd)
    root = Path(root_text.strip()).resolve() if root_text else Path(__file__).resolve().parents[2]
    commit = _git_output(["rev-parse", "HEAD"], root)
    status = _git_output(["status", "--porcelain"], root)
    diff = _git_output(
        ["diff", "--binary", "HEAD", "--", "src", "examples", "pyproject.toml"],
        root,
        binary=True,
    )
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = None
    return {
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(status and status.strip()),
        "tracked_source_diff_sha256": (
            hashlib.sha256(diff).hexdigest() if diff is not None else None
        ),
        "source_tree_sha256": _source_tree_sha256(root),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "platform": platform.platform(),
        "command": list(sys.argv),
        "working_directory": str(cwd.resolve()),
    }


def _history_payload(result: Any) -> dict[str, Any]:
    history = result.history if hasattr(result, "history") else result
    config = getattr(result, "config", None)
    payload: dict[str, Any] = {"history": _json_safe(history)}
    if config is not None:
        payload["effective_training_config"] = _json_safe(config)
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_training_history(
    results: Any | Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Persist raw training values and the effective training configuration."""

    if hasattr(results, "history"):
        payload: dict[str, Any] = _history_payload(results)
    elif isinstance(results, Mapping):
        payload = {
            "algorithms": {
                str(name): _history_payload(result)
                for name, result in results.items()
            }
        }
    else:
        raise TypeError("results must be a TrainingResult or a mapping of results")
    destination = Path(path)
    destination.write_text(_json_text(payload), encoding="utf-8")
    return destination


@dataclass
class RunArtifacts:
    """Paths and metadata belonging to one experiment execution."""

    example_name: str
    output_root: Path
    output_dir: Path
    run_id: str
    config_sha256: str
    source_sha256: str
    started_at: str

    def complete(self, *, metrics_file: str | None = "metrics.json") -> None:
        """Append the completed run and its metrics to the comparison index."""

        completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        index_record = {
            "run_id": self.run_id,
            "example": self.example_name,
            "status": "completed",
            "started_at": self.started_at,
            "completed_at": completed_at,
            "config_sha256": self.config_sha256,
            "source_sha256": self.source_sha256,
            "run_dir": self.output_dir.relative_to(self.output_root).as_posix(),
            "config_file": f"{self.run_id}/config.json",
            "metrics_file": (
                f"{self.run_id}/{metrics_file}" if metrics_file is not None else None
            ),
            "metrics": (
                json.loads((self.output_dir / metrics_file).read_text(encoding="utf-8"))
                if metrics_file is not None
                and (self.output_dir / metrics_file).is_file()
                else None
            ),
        }
        with (self.output_root / "runs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(index_record, sort_keys=True) + "\n")


def create_run_artifacts(
    config: Mapping[str, Any],
    output_root: str | Path,
    example_name: str,
    *,
    run_name: str | None = None,
) -> RunArtifacts:
    """Create a unique run directory and persist its exact effective config.

    ``run_name`` is an optional human-readable prefix. When omitted, the helper
    also checks ``config["run"]["name"]``. Every ID additionally contains its
    start time and a digest of the saved configuration.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_text = _json_text(config)
    config_sha256 = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    started = datetime.now().astimezone()
    configured_name = config.get("run", {}).get("name")
    label = run_name if run_name is not None else configured_name
    pieces = []
    if label:
        pieces.append(_slug(str(label)))
    pieces.extend(
        [
            started.strftime("%Y%m%d-%H%M%S"),
            config_sha256[:10],
        ]
    )
    base_id = "_".join(pieces)
    run_id = base_id
    suffix = 2
    while (output_root / run_id).exists():
        run_id = f"{base_id}_{suffix}"
        suffix += 1
    output_dir = output_root / run_id
    output_dir.mkdir()
    (output_dir / "config.json").write_text(config_text, encoding="utf-8")

    started_at = started.isoformat(timespec="seconds")
    provenance = _provenance(Path.cwd())
    source_sha256 = str(provenance["source_tree_sha256"])
    manifest = {
        "example": example_name,
        "run_id": run_id,
        "started_at": started_at,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "provenance": provenance,
    }
    (output_dir / "manifest.json").write_text(
        _json_text(manifest),
        encoding="utf-8",
    )
    print(f"Run {run_id}: artifacts will be written to {output_dir.resolve()}")
    return RunArtifacts(
        example_name=example_name,
        output_root=output_root,
        output_dir=output_dir,
        run_id=run_id,
        config_sha256=config_sha256,
        source_sha256=source_sha256,
        started_at=started_at,
    )
