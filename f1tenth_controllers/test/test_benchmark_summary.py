import json
from pathlib import Path

from f1tenth_controllers.benchmark.summary import (
    generate_summaries,
    summarize_head_to_head,
    summarize_time_trials,
)


def test_time_trial_summary_keeps_dnfs_out_of_lap_statistics() -> None:
    rows = [
        {
            "algorithm": "MATD3",
            "completion_status": "completed",
            "lap_time": "10.0",
            "dnf_reason": "",
        },
        {
            "algorithm": "MATD3",
            "completion_status": "completed",
            "lap_time": "12.0",
            "dnf_reason": "",
        },
        {
            "algorithm": "MATD3",
            "completion_status": "dnf",
            "lap_time": "",
            "dnf_reason": "collision",
        },
        {
            "algorithm": "ISAC",
            "completion_status": "dnf",
            "lap_time": "",
            "dnf_reason": "timeout",
        },
    ]

    summary = summarize_time_trials(rows)

    matd3 = summary["algorithms"]["MATD3"]
    assert matd3["completion_rate"] == 2 / 3
    assert matd3["dnf_counts"] == {"collision": 1}
    assert matd3["completed_lap_time"]["count"] == 2
    assert matd3["completed_lap_time"]["mean_sim_seconds"] == 11.0
    assert matd3["completed_lap_time"][
        "mean_95_percent_confidence_interval_sim_seconds"
    ] is not None
    assert summary["algorithms"]["ISAC"]["completed_lap_time"][
        "mean_sim_seconds"
    ] is None


def test_generate_summaries_reads_stable_csv_without_plots(
    tmp_path: Path,
) -> None:
    (tmp_path / "time_trial_trials.csv").write_text(
        "algorithm,completion_status,lap_time,dnf_reason\n"
        "ISAC,dnf,,timeout\n",
        encoding="utf-8",
    )
    (tmp_path / "head_to_head_trials.csv").write_text(
        "algorithm_a,algorithm_b,outcome_type,winner,lead_m\n"
        "ISAC,MATD3,crash_win,MATD3,2.5\n",
        encoding="utf-8",
    )

    summaries = generate_summaries(tmp_path, plots=False)

    assert summaries["time_trials"]["trial_count"] == 1
    assert summaries["head_to_head"]["heat_count"] == 1
    assert summaries["head_to_head"]["algorithms"]["MATD3"]["wins"] == 1
    assert summaries["head_to_head"]["algorithms"]["MATD3"][
        "winning_lead_m"
    ]["mean"] == 2.5
    written = json.loads(
        (tmp_path / "time_trial_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["algorithms"]["ISAC"]["dnf_counts"] == {"timeout": 1}


def test_head_to_head_summary_omits_missing_leads_from_statistics() -> None:
    summary = summarize_head_to_head(
        [
            {
                "algorithm_a": "ISAC",
                "algorithm_b": "MATD3",
                "outcome_type": "timeout_draw",
                "winner": "",
                "lead_m": "",
            }
        ]
    )

    assert summary["algorithms"]["ISAC"]["winning_lead_m"] == {
        "count": 0,
        "median": None,
        "mean": None,
        "minimum": None,
        "maximum": None,
    }
