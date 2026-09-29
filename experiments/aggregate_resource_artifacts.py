#!/usr/bin/env python3
"""Aggregate existing original-protocol stage metrics without rerunning models."""
import csv
import json
from pathlib import Path


def main(root: Path, output: Path) -> None:
    rows = []
    for path in sorted((root / "outputs" / "original_protocol").glob("*/*W2A2*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != "original_paper_v1":
            continue
        stages = payload.get("stage_metrics", {})
        row = {
            "arch": payload.get("arch"),
            "protocol": payload.get("protocol"),
            "paper_comparable": payload.get("paper_comparable"),
            "quantized_top1": payload.get("quantized_top1"),
            "calibration_samples": payload.get("calibration_count"),
            "validation_count": payload.get("validation_count"),
            "wall_time_seconds": payload.get("wall_time_seconds"),
            "peak_memory_allocated_mb": payload.get("peak_memory_mb"),
        }
        for name in ("hessian", "perturbation", "memory_bank", "reconstruction", "end_to_end_calibration"):
            stage = stages.get(name) or stages.get("routing", {}).get(name, {})
            row[f"{name}_wall_time_seconds"] = stage.get("wall_time_seconds") if isinstance(stage, dict) else None
            row[f"{name}_peak_memory_mb"] = stage.get("peak_memory_mb") if isinstance(stage, dict) else None
            row[f"{name}_flops"] = stage.get("flops") if isinstance(stage, dict) else None
        rows.append(row)
    if not rows:
        raise SystemExit("no original_paper_v1 summaries found")
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration_breakdown.json").write_text(json.dumps({"protocol": "original_paper_v1", "rows": rows}, indent=2), encoding="utf-8")
    with (output / "calibration_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(Path("/workspace/hma-dcc-ptq"), Path("/workspace/hma-dcc-ptq/experiments/ispa_gpu_eval"))
