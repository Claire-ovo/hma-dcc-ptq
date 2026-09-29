#!/usr/bin/env python3
"""Select pre-generated same-K random routes that match an HMA route's FLOPs."""

import argparse
import json
from pathlib import Path


def load_json(path: str):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def selected(route):
    value = route.get("selected") if isinstance(route, dict) else route
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise ValueError("Each route must be a string list or contain a string-list 'selected' field.")
    if len(value) != len(set(value)):
        raise ValueError("Route contains duplicate blocks.")
    return set(value)


def route_cost(block_costs, high_budget):
    missing = sorted(high_budget.difference(block_costs))
    if missing:
        raise ValueError("Route contains blocks absent from the cost profile: " + ", ".join(missing))
    return sum(
        costs["high_budget_flops"] if block in high_budget else costs["low_budget_flops"]
        for block, costs in block_costs.items()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-summary", required=True, help="C1 summary.json containing route_selections.")
    parser.add_argument("--profile-summary", required=True, help="HMA-DCC phase-2 summary.json containing route_cost_profile.")
    parser.add_argument("--hma-route-key", default="hma_same_k")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Maximum relative FLOPs difference.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not 0 <= args.tolerance < 1:
        raise ValueError("--tolerance must be in [0, 1).")

    metric_summary = load_json(args.metric_summary)
    profile_summary = load_json(args.profile_summary)
    if profile_summary.get("infonce_enabled") is not False:
        raise ValueError(
            "Compute-matched routing selection requires a profile generated with --disable-infonce. "
            "InfoNCE is evaluated separately with a fixed route and budget."
        )
    routes = metric_summary.get("route_selections")
    if not isinstance(routes, dict):
        raise ValueError("Metric summary must contain route_selections.")
    if args.hma_route_key not in routes:
        raise KeyError(f"HMA route {args.hma_route_key!r} not found.")

    block_costs = profile_summary.get("route_cost_profile")
    if not isinstance(block_costs, dict) or not block_costs:
        raise ValueError("Profile summary has no route_cost_profile. Run HMA-DCC with FLOPs profiling first.")
    required_fields = {"high_budget_flops", "low_budget_flops"}
    for block, costs in block_costs.items():
        if not isinstance(costs, dict) or not required_fields.issubset(costs):
            raise ValueError(f"Incomplete cost profile for block {block!r}.")
        if not all(isinstance(costs[field], (int, float)) and costs[field] >= 0 for field in required_fields):
            raise ValueError(f"Invalid numerical cost profile for block {block!r}.")

    hma_selected = selected(routes[args.hma_route_key])
    hma_cost = route_cost(block_costs, hma_selected)
    eligible, audit = {}, {}
    for route_name, route in sorted(routes.items()):
        if not route_name.startswith("random_"):
            continue
        candidate = selected(route)
        if len(candidate) != len(hma_selected):
            raise ValueError(f"{route_name} does not use HMA's K.")
        candidate_cost = route_cost(block_costs, candidate)
        relative_difference = abs(candidate_cost - hma_cost) / hma_cost if hma_cost else 0.0
        accepted = relative_difference <= args.tolerance
        audit[route_name] = {
            "selected": sorted(candidate),
            "estimated_flops": candidate_cost,
            "relative_difference": relative_difference,
            "accepted": accepted,
        }
        if accepted:
            eligible[route_name] = {"selected": sorted(candidate)}

    output = {
        "source": {
            "metric_summary": str(Path(args.metric_summary).resolve()),
            "profile_summary": str(Path(args.profile_summary).resolve()),
            "hma_route_key": args.hma_route_key,
        },
        "hma_reference": {
            "selected": sorted(hma_selected),
            "estimated_flops": hma_cost,
            "tolerance": args.tolerance,
        },
        "route_selections": eligible,
        "candidate_audit": audit,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"Accepted {len(eligible)}/{len(audit)} random routes within "
        f"{args.tolerance:.1%} of HMA's estimated FLOPs: {output_path}"
    )


if __name__ == "__main__":
    main()
