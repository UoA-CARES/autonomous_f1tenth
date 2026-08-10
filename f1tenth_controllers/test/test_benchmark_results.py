import csv
import json
from pathlib import Path

import pytest

from f1tenth_controllers.benchmark.results import (
    HEAD_TO_HEAD_FIELDS,
    TIME_TRIAL_FIELDS,
    ResultWriter,
    stable_id,
)


def test_stable_id_is_order_independent_and_content_sensitive() -> None:
    first = stable_id("trial", {"algorithm": "MATD3", "seed": 42})
    reordered = stable_id("trial", {"seed": 42, "algorithm": "MATD3"})
    changed = stable_id("trial", {"algorithm": "MATD3", "seed": 100})

    assert first == reordered
    assert first != changed


def test_result_writer_emits_stable_headers_and_json_fields(
    tmp_path: Path,
) -> None:
    writer = ResultWriter(tmp_path)
    trial_row = {
        field: None for field in TIME_TRIAL_FIELDS
    }
    trial_row.update(
        {
            "trial_id": "trial_1",
            "algorithm": "MATD3",
            "collision": False,
            "git_revisions": {"autonomous_f1tenth": "abc"},
        }
    )
    writer.write_time_trial(trial_row)

    race_row = {
        field: None for field in HEAD_TO_HEAD_FIELDS
    }
    race_row.update(
        {
            "heat_id": "heat_1",
            "algorithm_a": "MATD3",
            "algorithm_b": "MAPPO",
            "final_progress_m": {"MATD3": 40.0, "MAPPO": 39.0},
        }
    )
    writer.write_head_to_head(race_row)

    with writer.time_trial_path.open(newline="", encoding="utf-8") as result:
        trial = next(csv.DictReader(result))
    with writer.head_to_head_path.open(
        newline="", encoding="utf-8"
    ) as result:
        race = next(csv.DictReader(result))

    assert tuple(trial) == TIME_TRIAL_FIELDS
    assert tuple(race) == HEAD_TO_HEAD_FIELDS
    assert trial["collision"] == "0"
    assert json.loads(trial["git_revisions"]) == {
        "autonomous_f1tenth": "abc"
    }
    assert json.loads(race["final_progress_m"])["MAPPO"] == 39.0


def test_result_writer_rejects_duplicate_trial_and_heat_ids(
    tmp_path: Path,
) -> None:
    writer = ResultWriter(tmp_path)
    trial = {"trial_id": "trial_1"}
    heat = {"heat_id": "heat_1"}

    writer.write_time_trial(trial)
    writer.write_head_to_head(heat)

    with pytest.raises(ValueError, match="Duplicate time-trial"):
        writer.write_time_trial(trial)
    with pytest.raises(ValueError, match="Duplicate heat"):
        writer.write_head_to_head(heat)


def test_manifest_is_stable_and_cannot_be_replaced(tmp_path: Path) -> None:
    writer = ResultWriter(tmp_path)
    manifest = writer.write_manifest(
        {"config_id": "abc", "checkpoints": {"MATD3": "hash"}}
    )

    assert manifest["manifest_id"].startswith("manifest_")
    assert writer.write_manifest(
        {"config_id": "abc", "checkpoints": {"MATD3": "hash"}}
    ) == manifest
    reopened = writer.write_manifest(
        {
            "created_at": "later",
            "config_id": "abc",
            "checkpoints": {"MATD3": "hash"},
        }
    )
    assert reopened == manifest

    with pytest.raises(ValueError, match="different manifest"):
        writer.write_manifest(
            {"config_id": "different", "checkpoints": {"MATD3": "hash"}}
        )


def test_result_writer_reports_existing_ids_after_reopen(
    tmp_path: Path,
) -> None:
    writer = ResultWriter(tmp_path)
    writer.write_time_trial({"trial_id": "trial_1"})
    writer.write_head_to_head({"heat_id": "heat_1"})

    reopened = ResultWriter(tmp_path)

    assert reopened.has_time_trial("trial_1")
    assert reopened.has_head_to_head("heat_1")
    assert not reopened.has_time_trial("trial_missing")
    assert not reopened.has_head_to_head("heat_missing")
