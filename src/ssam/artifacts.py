"""Reproducible, non-overwriting artifact directories for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=lambda item: str(item),
    ) + "\n"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "run"


@dataclass
class RunArtifacts:
    """Paths and metadata belonging to one experiment execution."""

    example_name: str
    output_root: Path
    output_dir: Path
    run_id: str
    config_sha256: str
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
    print(f"Run {run_id}: artifacts will be written to {output_dir.resolve()}")
    return RunArtifacts(
        example_name=example_name,
        output_root=output_root,
        output_dir=output_dir,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )
