#!/usr/bin/env python3
"""
Re-evaluate PatternMemoryAnalyzer results using only benchmark cases that
contain at least one expected pattern.

Matching is reused from the original benchmark result file's
pairwise_similarity[].accepted_match values. This script does not call an
LLM, reload embeddings, or change the original result/dataset files.

For included (positive) cases:
  TP = accepted semantic matches
  FP = unmatched predicted patterns
  FN = unmatched expected patterns

Cases whose dataset expected.patterns is [] are excluded completely, including
any predictions made for those cases.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def f1_score(precision, recall):
    return safe_div(2 * precision * recall, precision + recall)


def accepted_match_count(case_result):
    """Count unique accepted prediction↔expected pairs."""
    accepted = set()
    for pair in case_result.get("pairwise_similarity", []) or []:
        if pair.get("accepted_match") is True:
            accepted.add((pair.get("predicted_index"), pair.get("expected_index")))
    return len(accepted)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate only cases with non-empty expected patterns."
    )
    parser.add_argument(
        "--results",
        default="benchmark/results/pattern_memory_benchmark_results.json",
        help="Original analyzer benchmark results JSON",
    )
    parser.add_argument(
        "--dataset",
        default="benchmark/datasets/pattern_benchmark.json",
        help="Original benchmark dataset JSON",
    )
    parser.add_argument(
        "--output",
        default="benchmark/results/pattern_memory_positive_only_evaluation.json",
        help="Where to save the filtered evaluation report",
    )
    args = parser.parse_args()

    results = load_json(args.results)
    dataset = load_json(args.dataset)

    expected_by_id = {
        case["case_id"]: case.get("expected", {}).get("patterns", []) or []
        for case in dataset.get("cases", [])
    }
    result_by_id = {case["case_id"]: case for case in results.get("cases", [])}

    positive_ids = {
        case_id for case_id, expected in expected_by_id.items() if len(expected) > 0
    }
    ignored_empty_ids = set(expected_by_id) - positive_ids

    totals = defaultdict(int)
    scenario_totals = defaultdict(lambda: defaultdict(int))
    per_case = []
    failed_positive = []
    missing_result = []

    evidence_quote_total = 0
    evidence_quote_valid = 0
    evidence_episode_coverage_values = []
    lifecycle_values = []
    pattern_id_values = []
    duplicate_pairs = 0

    for case_id in sorted(positive_ids):
        expected = expected_by_id[case_id]
        result = result_by_id.get(case_id)

        if result is None:
            missing_result.append(case_id)
            continue

        if result.get("failed") or result.get("error"):
            failed_positive.append({
                "case_id": case_id,
                "error": result.get("error", "Marked failed"),
            })
            continue

        predictions = result.get("predictions", []) or []
        tp = accepted_match_count(result)

        # Protect against malformed/stale pairwise match data.
        tp = min(tp, len(predictions), len(expected))
        fp = max(0, len(predictions) - tp)
        fn = max(0, len(expected) - tp)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        case_f1 = f1_score(precision, recall)

        scenario = result.get("scenario", "unknown")
        for key, value in (("tp", tp), ("fp", fp), ("fn", fn),
                           ("predicted", len(predictions)), ("expected", len(expected)),
                           ("cases", 1)):
            totals[key] += value
            scenario_totals[scenario][key] += value

        evidence_quote_total += int(result.get("evidence_quotes_total") or 0)
        evidence_quote_valid += int(result.get("evidence_quotes_valid") or 0)

        coverage = result.get("evidence_episode_coverage")
        if coverage is not None:
            evidence_episode_coverage_values.append(float(coverage))

        lifecycle = result.get("lifecycle_accuracy")
        if lifecycle is not None:
            lifecycle_values.append(float(lifecycle))

        pattern_id = result.get("pattern_id_accuracy")
        if pattern_id is not None:
            pattern_id_values.append(float(pattern_id))

        duplicate_pairs += int(result.get("duplicate_prediction_pairs") or 0)

        per_case.append({
            "case_id": case_id,
            "scenario": scenario,
            "expected_count": len(expected),
            "predicted_count": len(predictions),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(case_f1, 4),
        })

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    micro_precision = safe_div(tp, tp + fp)
    micro_recall = safe_div(tp, tp + fn)
    micro_f1 = f1_score(micro_precision, micro_recall)

    macro_case_f1 = safe_div(
        sum(case["f1"] for case in per_case), len(per_case)
    )

    by_scenario = {}
    for scenario, values in sorted(scenario_totals.items()):
        p = safe_div(values["tp"], values["tp"] + values["fp"])
        r = safe_div(values["tp"], values["tp"] + values["fn"])
        by_scenario[scenario] = {
            "cases": values["cases"],
            "expected_patterns": values["expected"],
            "predicted_patterns": values["predicted"],
            "tp": values["tp"],
            "fp": values["fp"],
            "fn": values["fn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1_score(p, r), 4),
        }

    report = {
        "evaluation": "positive_expected_patterns_only",
        "matching_source": "pairwise_similarity[].accepted_match from original results",
        "filtering": {
            "dataset_cases_total": len(expected_by_id),
            "cases_with_expected_patterns": len(positive_ids),
            "cases_ignored_expected_empty": len(ignored_empty_ids),
            "ignored_empty_case_ids": sorted(ignored_empty_ids),
            "positive_cases_completed_and_scored": len(per_case),
            "positive_cases_failed": len(failed_positive),
            "positive_cases_missing_from_results": len(missing_result),
        },
        "metrics": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "micro_precision": round(micro_precision, 4),
            "micro_recall": round(micro_recall, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_case_f1": round(macro_case_f1, 4),
            "evidence_quote_validity": round(
                safe_div(evidence_quote_valid, evidence_quote_total), 4
            ),
            "evidence_quotes_valid": evidence_quote_valid,
            "evidence_quotes_total": evidence_quote_total,
            "mean_evidence_episode_coverage": round(
                safe_div(
                    sum(evidence_episode_coverage_values),
                    len(evidence_episode_coverage_values),
                ), 4
            ),
            "mean_lifecycle_accuracy": round(
                safe_div(sum(lifecycle_values), len(lifecycle_values)), 4
            ) if lifecycle_values else None,
            "mean_pattern_id_accuracy": round(
                safe_div(sum(pattern_id_values), len(pattern_id_values)), 4
            ) if pattern_id_values else None,
            "duplicate_prediction_pairs": duplicate_pairs,
        },
        "by_scenario": by_scenario,
        "failed_positive_cases": failed_positive,
        "missing_positive_cases": missing_result,
        "per_case": per_case,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    print("\n========== POSITIVE-EXPECTED-PATTERNS BENCHMARK ==========")
    print(f"Dataset cases: {len(expected_by_id)}")
    print(f"Cases with expected patterns: {len(positive_ids)}")
    print(f"Ignored empty-expected cases: {len(ignored_empty_ids)}")
    print(f"Completed and scored: {len(per_case)}")
    print(f"Failed positive cases: {len(failed_positive)}")
    print(f"Missing positive cases: {len(missing_result)}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall:    {micro_recall:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")
    print(f"Macro Case F1:   {macro_case_f1:.4f}")
    print(f"Evidence validity: {report['metrics']['evidence_quote_validity']:.4f}")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
