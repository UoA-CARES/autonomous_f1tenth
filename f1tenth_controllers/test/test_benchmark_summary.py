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
            "checkpoint_id": "MATD3_seed",
            "algorithm": "MATD3",
            "checkpoint_filename": "MATD3_seed.pth",
            "completion_status": "completed",
            "lap_time": "10.0",
            "dnf_reason": "",
        },
        {
            "checkpoint_id": "MATD3_seed",
            "algorithm": "MATD3",
            "checkpoint_filename": "MATD3_seed.pth",
            "completion_status": "completed",
            "lap_time": "12.0",
            "dnf_reason": "",
        },
        {
            "checkpoint_id": "MATD3_seed",
            "algorithm": "MATD3",
            "checkpoint_filename": "MATD3_seed.pth",
            "completion_status": "dnf",
            "lap_time": "",
            "dnf_reason": "collision",
        },
        {
            "checkpoint_id": "ISAC_seed",
            "algorithm": "ISAC",
            "checkpoint_filename": "ISAC_seed.pth",
            "completion_status": "dnf",
            "lap_time": "",
            "dnf_reason": "timeout",
        },
    ]

    summary = summarize_time_trials(rows)

    matd3 = summary["competitors"]["MATD3_seed"]
    assert matd3["completion_rate"] == 2 / 3
    assert matd3["dnf_counts"] == {"collision": 1}
    assert matd3["completed_lap_time"]["count"] == 2
    assert matd3["completed_lap_time"]["mean_sim_seconds"] == 11.0
    assert matd3["completed_lap_time"][
        "mean_95_percent_confidence_interval_sim_seconds"
    ] is not None
    assert summary["competitors"]["ISAC_seed"]["completed_lap_time"][
        "mean_sim_seconds"
    ] is None


def test_generate_summaries_reads_stable_csv_without_plots(
    tmp_path: Path,
) -> None:
    (tmp_path / "time_trial_trials.csv").write_text(
        "checkpoint_id,algorithm,checkpoint_filename,completion_status,lap_time,dnf_reason\n"
        "ISAC_seed,ISAC,ISAC_seed.pth,dnf,,timeout\n",
        encoding="utf-8",
    )
    (tmp_path / "head_to_head_trials.csv").write_text(
        "checkpoint_id_a,algorithm_a,checkpoint_id_b,algorithm_b,"
        "outcome_type,winner_checkpoint_id,lead_m\n"
        "ISAC_seed,ISAC,MATD3_seed,MATD3,crash_win,MATD3_seed,2.5\n",
        encoding="utf-8",
    )

    summaries = generate_summaries(tmp_path, plots=False)

    assert summaries["time_trials"]["trial_count"] == 1
    assert summaries["head_to_head"]["heat_count"] == 1
    assert summaries["head_to_head"]["competitors"]["MATD3_seed"]["wins"] == 1
    assert summaries["head_to_head"]["competitors"]["MATD3_seed"][
        "winning_lead_m"
    ]["mean"] == 2.5
    written = json.loads(
        (tmp_path / "time_trial_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["competitors"]["ISAC_seed"]["dnf_counts"] == {"timeout": 1}


def test_head_to_head_summary_omits_missing_leads_from_statistics() -> None:
    summary = summarize_head_to_head(
        [
            {
                "checkpoint_id_a": "ISAC_seed",
                "algorithm_a": "ISAC",
                "checkpoint_id_b": "MATD3_seed",
                "algorithm_b": "MATD3",
                "outcome_type": "timeout_draw",
                "winner_checkpoint_id": "",
                "lead_m": "",
            }
        ]
    )

    assert summary["competitors"]["ISAC_seed"]["winning_lead_m"] == {
        "count": 0,
        "median": None,
        "mean": None,
        "minimum": None,
        "maximum": None,
    }
