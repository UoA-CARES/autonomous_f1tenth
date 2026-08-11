"""Stable machine-readable result artifacts for MARL benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


TIME_TRIAL_FIELDS = (
    "trial_id",
    "checkpoint_id",
    "algorithm",
    "checkpoint_filename",
    "checkpoint_sha256",
    "actor_id",
    "track",
    "direction",
    "seed",
    "start_sim_time",
    "finish_sim_time",
    "lap_time",
    "completion_status",
    "dnf_reason",
    "distance_completed_m",
    "average_speed_mps",
    "maximum_speed_mps",
    "collision",
    "flip",
    "stall",
    "timeout",
    "git_revisions",
    "manifest_id",
    "config_id",
)

HEAD_TO_HEAD_FIELDS = (
    "heat_id",
    "checkpoint_id_a",
    "algorithm_a",
    "checkpoint_a",
    "checkpoint_sha256_a",
    "checkpoint_id_b",
    "algorithm_b",
    "checkpoint_b",
    "checkpoint_sha256_b",
    "lead_checkpoint_id",
    "lead_algorithm",
    "chaser_checkpoint_id",
    "chaser_algorithm",
    "primary_checkpoint_id",
    "primary_algorithm",
    "opponent_checkpoint_id",
    "opponent_algorithm",
    "track",
    "direction",
    "seed",
    "start_separation_m",
    "outcome_type",
    "winner_checkpoint_id",
    "winner_algorithm",
    "finish_or_crash_sim_time",
    "lead_m",
    "final_progress_m",
    "crash_participant_checkpoint_ids",
    "crash_participant_algorithms",
    "responsible_checkpoint_id",
    "responsible_algorithm",
    "attribution_confidence",
    "attribution_evidence",
    "dnf_reasons",
    "git_revisions",
    "manifest_id",
    "config_id",
)


def stable_id(prefix: str, payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:20]}"


def manifest_id(manifest: dict) -> str:
    stable_content = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_id", "created_at"}
    }
    return stable_id("manifest", stable_content)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return int(value)
    return value


class ResultWriter:
    """Append exactly one row per unique trial/heat and emit JSONL events."""

    def __init__(self, result_directory: Path) -> None:
        self.result_directory = Path(result_directory)
        self.result_directory.mkdir(parents=True, exist_ok=True)
        self.time_trial_path = (
            self.result_directory / "time_trial_trials.csv"
        )
        self.head_to_head_path = (
            self.result_directory / "head_to_head_trials.csv"
        )
        self.events_path = self.result_directory / "events.jsonl"
        self._time_trial_ids = self._existing_ids(
            self.time_trial_path, "trial_id"
        )
        self._heat_ids = self._existing_ids(
            self.head_to_head_path, "heat_id"
        )

    @staticmethod
    def _existing_ids(path: Path, id_field: str) -> set[str]:
        if not path.is_file() or path.stat().st_size == 0:
            return set()
        with path.open(newline="", encoding="utf-8") as result_file:
            return {
                row[id_field]
                for row in csv.DictReader(result_file)
                if row.get(id_field)
            }

    @staticmethod
    def _append_csv(path: Path, fields: tuple[str, ...], row: dict) -> None:
        unexpected = set(row) - set(fields)
        if unexpected:
            raise ValueError(
                f"Unexpected result fields for {path.name}: "
                f"{sorted(unexpected)}"
            )
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as result_file:
            writer = csv.DictWriter(result_file, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {field: _csv_value(row.get(field)) for field in fields}
            )

    def write_manifest(self, manifest: dict) -> dict:
        resolved_manifest = dict(manifest)
        resolved_manifest["manifest_id"] = manifest_id(resolved_manifest)
        target = self.result_directory / "run_manifest.json"
        if target.exists():
            with target.open(encoding="utf-8") as manifest_file:
                existing = json.load(manifest_file)
            if existing.get("manifest_id") != resolved_manifest["manifest_id"]:
                raise ValueError(
                    "Result directory already contains a different manifest"
                )
            return existing
        with target.open("x", encoding="utf-8") as manifest_file:
            json.dump(
                resolved_manifest,
                manifest_file,
                indent=2,
                sort_keys=True,
            )
            manifest_file.write("\n")
        return resolved_manifest

    def has_time_trial(self, trial_id: str) -> bool:
        return trial_id in self._time_trial_ids

    def has_head_to_head(self, heat_id: str) -> bool:
        return heat_id in self._heat_ids

    def write_time_trial(self, row: dict) -> None:
        trial_id = str(row["trial_id"])
        if trial_id in self._time_trial_ids:
            raise ValueError(f"Duplicate time-trial ID {trial_id}")
        self._append_csv(self.time_trial_path, TIME_TRIAL_FIELDS, row)
        self._time_trial_ids.add(trial_id)

    def write_head_to_head(self, row: dict) -> None:
        heat_id = str(row["heat_id"])
        if heat_id in self._heat_ids:
            raise ValueError(f"Duplicate heat ID {heat_id}")
        self._append_csv(self.head_to_head_path, HEAD_TO_HEAD_FIELDS, row)
        self._heat_ids.add(heat_id)

    def write_event(self, event: dict) -> None:
        with self.events_path.open("a", encoding="utf-8") as events_file:
            events_file.write(
                json.dumps(event, sort_keys=True, separators=(",", ":"))
            )
            events_file.write("\n")
