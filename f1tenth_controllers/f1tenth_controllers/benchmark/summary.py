"""Regenerate benchmark summaries and plots from stable CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import t as student_t


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as result_file:
        return list(csv.DictReader(result_file))


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def _completed_lap_statistics(lap_times: list[float]) -> dict:
    count = len(lap_times)
    if count == 0:
        return {
            "count": 0,
            "median_sim_seconds": None,
            "mean_sim_seconds": None,
            "sample_standard_deviation_seconds": None,
            "mean_95_percent_confidence_interval_sim_seconds": None,
            "confidence_interval_method": "two_sided_student_t",
        }

    mean = statistics.fmean(lap_times)
    standard_deviation = (
        statistics.stdev(lap_times) if count >= 2 else None
    )
    confidence_interval = None
    if standard_deviation is not None:
        critical = float(student_t.ppf(0.975, df=count - 1))
        margin = critical * standard_deviation / math.sqrt(count)
        confidence_interval = [mean - margin, mean + margin]

    return {
        "count": count,
        "median_sim_seconds": statistics.median(lap_times),
        "mean_sim_seconds": mean,
        "sample_standard_deviation_seconds": standard_deviation,
        "mean_95_percent_confidence_interval_sim_seconds": (
            confidence_interval
        ),
        "confidence_interval_method": "two_sided_student_t",
    }


def summarize_time_trials(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["checkpoint_id"]].append(row)

    competitors = {}
    for checkpoint_id in sorted(grouped):
        trials = grouped[checkpoint_id]
        completed = [
            row
            for row in trials
            if row["completion_status"] == "completed"
        ]
        lap_times = [
            float(row["lap_time"])
            for row in completed
            if row["lap_time"] != ""
        ]
        dnf_counts = Counter(
            row["dnf_reason"] or "unspecified"
            for row in trials
            if row["completion_status"] != "completed"
        )
        competitors[checkpoint_id] = {
            "algorithm": trials[0]["algorithm"],
            "checkpoint_filename": trials[0]["checkpoint_filename"],
            "trial_count": len(trials),
            "valid_lap_count": len(completed),
            "completion_rate": len(completed) / len(trials),
            "dnf_counts": dict(sorted(dnf_counts.items())),
            "completed_lap_time": _completed_lap_statistics(lap_times),
        }
    return {
        "schema_version": 2,
        "trial_count": len(rows),
        "competitors": competitors,
        "note": (
            "DNFs are excluded from lap-time statistics; completion rate "
            "reports reliability separately."
        ),
    }


def summarize_head_to_head(rows: list[dict]) -> dict:
    outcome_counts = Counter(row["outcome_type"] for row in rows)
    competitors = defaultdict(
        lambda: {
            "starts": 0,
            "wins": 0,
            "finish_wins": 0,
            "crash_wins": 0,
        }
    )
    winning_leads = defaultdict(list)
    pairings = defaultdict(list)
    for row in rows:
        checkpoint_a = row["checkpoint_id_a"]
        checkpoint_b = row["checkpoint_id_b"]
        competitors[checkpoint_a]["algorithm"] = row["algorithm_a"]
        competitors[checkpoint_b]["algorithm"] = row["algorithm_b"]
        competitors[checkpoint_a]["starts"] += 1
        competitors[checkpoint_b]["starts"] += 1
        winner = row["winner_checkpoint_id"]
        if winner:
            competitors[winner]["wins"] += 1
            if row["lead_m"] != "":
                winning_leads[winner].append(float(row["lead_m"]))
            if row["outcome_type"] == "finish_win":
                competitors[winner]["finish_wins"] += 1
            elif row["outcome_type"] == "crash_win":
                competitors[winner]["crash_wins"] += 1
        pairing = tuple(sorted((checkpoint_a, checkpoint_b)))
        pairings[pairing].append(row)

    competitor_summary = {}
    for checkpoint_id, values in sorted(competitors.items()):
        leads = winning_leads[checkpoint_id]
        competitor_summary[checkpoint_id] = {
            **values,
            "win_rate": (
                values["wins"] / values["starts"]
                if values["starts"]
                else 0.0
            ),
            "winning_lead_m": {
                "count": len(leads),
                "median": statistics.median(leads) if leads else None,
                "mean": statistics.fmean(leads) if leads else None,
                "minimum": min(leads) if leads else None,
                "maximum": max(leads) if leads else None,
            },
        }

    pairing_summary = {}
    for pairing, pairing_rows in sorted(pairings.items()):
        pairing_key = f"{pairing[0]}__vs__{pairing[1]}"
        pairing_summary[pairing_key] = {
            "heat_count": len(pairing_rows),
            "outcome_counts": dict(
                sorted(
                    Counter(
                        row["outcome_type"] for row in pairing_rows
                    ).items()
                )
            ),
            "winner_counts": dict(
                sorted(
                    Counter(
                        row["winner_checkpoint_id"] or "none" for row in pairing_rows
                    ).items()
                )
            ),
        }

    return {
        "schema_version": 2,
        "heat_count": len(rows),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "competitors": competitor_summary,
        "pairings": pairing_summary,
    }


def _plot_time_trials(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    competitors = list(summary["competitors"])
    completion_rates = [
        summary["competitors"][competitor]["completion_rate"]
        for competitor in competitors
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(competitors, completion_rates)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Valid-lap completion rate")
    axis.set_title("MARL time-trial reliability")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_head_to_head(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    competitors = list(summary["competitors"])
    wins = [
        summary["competitors"][competitor]["wins"]
        for competitor in competitors
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(competitors, wins)
    axis.set_ylabel("Pilot/full heat wins")
    axis.set_title("MARL head-to-head wins")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate_summaries(
    result_directory: Path,
    *,
    plots: bool = True,
) -> dict[str, dict]:
    result_directory = Path(result_directory)
    time_trial_rows = _read_csv(
        result_directory / "time_trial_trials.csv"
    )
    head_to_head_rows = _read_csv(
        result_directory / "head_to_head_trials.csv"
    )
    summaries = {}
    if time_trial_rows:
        summary = summarize_time_trials(time_trial_rows)
        _write_json(
            result_directory / "time_trial_summary.json",
            summary,
        )
        if plots:
            _plot_time_trials(
                summary,
                result_directory / "time_trial_summary.png",
            )
        summaries["time_trials"] = summary
    if head_to_head_rows:
        summary = summarize_head_to_head(head_to_head_rows)
        _write_json(
            result_directory / "head_to_head_summary.json",
            summary,
        )
        if plots:
            _plot_head_to_head(
                summary,
                result_directory / "head_to_head_summary.png",
            )
        summaries["head_to_head"] = summary
    if not summaries:
        raise FileNotFoundError(
            f"No benchmark CSV artifacts found in {result_directory}"
        )
    return summaries


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate MARL benchmark summaries and plots"
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args(argv)
    summaries = generate_summaries(
        arguments.result_dir,
        plots=not arguments.no_plots,
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))
