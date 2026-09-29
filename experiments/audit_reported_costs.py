#!/usr/bin/env python3
"""Audit arithmetic and scope consistency of the reported calibration costs."""

import argparse
import json
from pathlib import Path


REPORTED = {
    "fixed_20k_tflops": 148.24,
    "fixed_10k_tflops": 74.12,
    "hma_dynamic_tflops": 37.06,
    "s2_no_infonce_tflops": 8.64,
    "s2_hard_head_off_tflops": 37.06,
    "s2_hard_head_on_tflops": 37.06,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fixed_5k_implied = REPORTED["fixed_20k_tflops"] * 5_000 / 20_000
    result = {
        "reported": REPORTED,
        "derived": {
            "fixed_20k_over_fixed_10k": REPORTED["fixed_20k_tflops"] / REPORTED["fixed_10k_tflops"],
            "implied_fixed_5k_tflops": fixed_5k_implied,
            "dynamic_over_implied_fixed_5k": REPORTED["hma_dynamic_tflops"] / fixed_5k_implied,
            "s2_no_infonce_over_dynamic": REPORTED["s2_no_infonce_tflops"] / REPORTED["hma_dynamic_tflops"],
        },
        "logical_check": {
            "assumptions": [
                "The same positive per-block FLOP weights and the same counted modules are used.",
                "Every robust block receives 5k iterations.",
                "At least one sensitive block receives 20k iterations.",
            ],
            "implication": "Dynamic 20k/5k cost must be strictly greater than all-block fixed 5k cost.",
            "observed": "Reported dynamic cost equals the fixed-5k cost implied by the linear fixed-20k/fixed-10k rows.",
            "status": "scope_or_accounting_mismatch_requires_raw_log_audit",
        },
        "source_code_scope": {
            "phase1_hessian_in_counter": False,
            "phase1_perturbation_in_counter": False,
            "phase1_in_wall_time": False,
            "phase1_in_peak_memory_window": False,
            "memory_bank_profiled_in_phase2_formula": True,
            "note": "main_hmadcc.py resets FLOPs/memory and starts timing after extract_hybrid_metrics().",
        },
        "unresolved_definitions": [
            "Main Table 4 Uniform Allocation (22.16%) is not the Supplement fixed-10k row (52.67%).",
            "The current public entry does not expose historical metric_type/allocation_type/infonce_type flags.",
            "The 8.64-TFLOP no-InfoNCE row cannot be reconstructed from the current public command path without historical logs/scripts.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
