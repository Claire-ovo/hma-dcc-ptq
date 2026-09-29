#!/usr/bin/env python3
"""Recompute correlations and exact same-K routing statistics from block metrics."""

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rebuttal_metrics import kendall_tau_b, spearman


SCORES = [
    "hessian_legacy_max",
    "hessian_block_trace",
    "hessian_abs_sum",
    "hessian_avg",
    "perturbation_mse",
    "hma_original",
    "hma_avg",
    "fusion_mean",
    "fusion_h75_p25",
    "fusion_h25_p75",
    "fusion_rank_sum",
]


def top_k(rows, score, k, reverse=True):
    return sorted(rows, key=lambda row: (float(row[score]), row["block"]), reverse=reverse)[:k]


def coverage(selected, total_positive):
    captured = sum(max(float(row["loss_increase"]), 0.0) for row in selected)
    return captured / total_positive if total_positive else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-head", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.metrics_csv).open()))
    if args.exclude_head:
        rows = [
            row
            for row in rows
            if not any(part in {"fc", "classifier"} for part in row["block"].split("."))
        ]
    source_summary = json.loads(Path(args.summary_json).read_text())
    threshold = float(source_summary["protocol"]["hma_threshold"])
    threshold_selected = [row for row in rows if float(row["hma_original"]) >= threshold]
    k = len(threshold_selected) or 1
    degradation = [float(row["loss_increase"]) for row in rows]
    total_positive = sum(max(value, 0.0) for value in degradation)

    correlations = {}
    routing = {}
    for score in SCORES:
        values = [float(row[score]) for row in rows]
        correlations[score] = {
            "spearman": spearman(values, degradation),
            "kendall": kendall_tau_b(values, degradation),
        }
        chosen = top_k(rows, score, k)
        routing[score] = {
            "coverage": coverage(chosen, total_positive),
            "selected": [row["block"] for row in chosen],
        }

    metric_disagreement = {}
    for hessian_name in ["hessian_legacy_max", "hessian_block_trace", "hessian_avg"]:
        hessian_values = [float(row[hessian_name]) for row in rows]
        perturbation_values = [float(row["perturbation_mse"]) for row in rows]
        metric_disagreement[hessian_name + "_vs_perturbation"] = {
            "spearman": spearman(hessian_values, perturbation_values),
            "kendall": kendall_tau_b(hessian_values, perturbation_values),
        }

    inverted = top_k(rows, "hma_original", k, reverse=False)
    routing["inverted_hma"] = {
        "coverage": coverage(inverted, total_positive),
        "selected": [row["block"] for row in inverted],
    }

    combinations = math.comb(len(rows), k)
    if combinations <= 1_000_000:
        random_coverages = [coverage(combo, total_positive) for combo in itertools.combinations(rows, k)]
        random_mode = "exact_all_combinations"
    else:
        raise RuntimeError(f"Exact enumeration too large: C({len(rows)}, {k})={combinations}")
    random_coverages.sort()
    hma_coverage = routing["hma_original"]["coverage"]
    no_better = sum(value <= hma_coverage + 1e-15 for value in random_coverages)
    at_least_as_good = sum(value >= hma_coverage - 1e-15 for value in random_coverages)

    result = {
        "source": args.metrics_csv,
        "exclude_head": args.exclude_head,
        "unit_count": len(rows),
        "k": k,
        "correlations": correlations,
        "metric_disagreement": metric_disagreement,
        "routing": routing,
        "random_same_k": {
            "mode": random_mode,
            "combination_count": combinations,
            "mean_coverage": sum(random_coverages) / len(random_coverages),
            "min_coverage": random_coverages[0],
            "max_coverage": random_coverages[-1],
            "hma_percentile": no_better / len(random_coverages),
            "fraction_random_at_least_as_good_as_hma": at_least_as_good / len(random_coverages),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
