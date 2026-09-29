from __future__ import annotations

import json

from ssam import create_run_artifacts


def test_run_artifacts_save_config_and_index(tmp_path):
    config = {"run": {"name": "PDE only"}, "training": {"seed": 17}}

    run = create_run_artifacts(config, tmp_path, "poisson_pinn")
    (run.output_dir / "metrics.json").write_text('{"loss": 1.0}\n', encoding="utf-8")
    (run.output_dir / "plot.png").write_bytes(b"figure")
    run.complete()

    assert run.run_id.startswith("PDE-only_")
    assert json.loads((run.output_dir / "config.json").read_text()) == config
    assert not (run.output_dir / "manifest.json").exists()
    records = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert records == [
        {
            "completed_at": records[0]["completed_at"],
            "config_file": f"{run.run_id}/config.json",
            "config_sha256": run.config_sha256,
            "example": "poisson_pinn",
            "metrics_file": f"{run.run_id}/metrics.json",
            "metrics": {"loss": 1.0},
            "run_dir": run.run_id,
            "run_id": run.run_id,
            "started_at": run.started_at,
            "status": "completed",
        }
    ]


def test_run_artifacts_never_overwrite_same_second(tmp_path):
    config = {"training": {"seed": 3}}
    first = create_run_artifacts(config, tmp_path, "example")
    second = create_run_artifacts(config, tmp_path, "example")

    assert first.output_dir != second.output_dir
    assert (first.output_dir / "config.json").is_file()
    assert (second.output_dir / "config.json").is_file()
