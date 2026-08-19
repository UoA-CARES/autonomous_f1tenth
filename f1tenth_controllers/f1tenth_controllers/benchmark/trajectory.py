"""Headless XY trajectory plots for benchmark trials and heats.

Colours follow the CARES/Claude dataviz validated default palette
(references/palette.md): categorical slots for per-car identity in
head-to-head plots, and the fixed status pair (good/critical) for
completed-vs-DNF runs in per-checkpoint aggregate plots, since completion is
a pass/fail state rather than an arbitrary series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .track_boundary import load_track_boundary_segments

# Categorical slots 1-8 (blue, orange, aqua, yellow, magenta, green, violet,
# red), fixed order - never cycled or reassigned by rank.
_CAR_COLOURS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

_CHART_SURFACE = "#fcfcfb"
_PRIMARY_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"
_GRIDLINE = "#e1e0d9"
_BOUNDARY_INK = "#52514e"

# Status pair: completion is a pass/fail state, not a categorical series.
_STATUS_GOOD = "#0ca30c"
_STATUS_CRITICAL = "#d03b3b"


def _style_axes(axes) -> None:
    axes.set_facecolor(_CHART_SURFACE)
    axes.set_xlabel("world X (m)", color=_SECONDARY_INK)
    axes.set_ylabel("world Y (m)", color=_SECONDARY_INK)
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True, linewidth=1.0, color=_GRIDLINE, zorder=0)
    axes.set_axisbelow(True)
    axes.tick_params(colors=_MUTED_INK, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(_GRIDLINE)


def _draw_track(axes, track_waypoints, track_name: str | None) -> list:
    """Draw the wall boundary (if resolvable) and the centreline. Returns legend handles."""
    handles = []

    if track_name is not None:
        boundary_segments = load_track_boundary_segments(track_name)
        if boundary_segments is not None:
            axes.add_collection(
                LineCollection(
                    boundary_segments,
                    colors=_BOUNDARY_INK,
                    linewidths=1.2,
                    alpha=0.9,
                    zorder=1,
                )
            )
            handles.append(
                Line2D([0], [0], color=_BOUNDARY_INK, linewidth=1.2, label="track boundary")
            )

    track = np.asarray(track_waypoints, dtype=np.float64)
    if track.ndim == 2 and track.shape[0] > 0 and track.shape[1] >= 2:
        track_xy = track[:, :2]
        if track_xy.shape[0] > 1:
            track_xy = np.vstack((track_xy, track_xy[0]))
        (line,) = axes.plot(
            track_xy[:, 0],
            track_xy[:, 1],
            color=_MUTED_INK,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="centreline",
            zorder=2,
        )
        handles.append(line)

    return handles


def write_xy_trajectory_plot(
    output_path: Path,
    *,
    track_waypoints,
    trajectories: dict[str, list[tuple[float, float]]],
    labels: dict[str, str] | None = None,
    title: str,
    track_name: str | None = None,
) -> Path:
    """Render the track and one sampled XY path per car to a PNG.

    Used for individual head-to-head heats, where exactly one specific
    matchup between (usually) two cars is the whole story.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(9.0, 7.0), constrained_layout=True, facecolor=_CHART_SURFACE)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)

    handles = _draw_track(axes, track_waypoints, track_name)

    labels = labels or {}
    for index, (agent, samples) in enumerate(trajectories.items()):
        path = np.asarray(samples, dtype=np.float64)
        if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
            continue
        path = path[:, :2]
        finite = np.isfinite(path).all(axis=1)
        path = path[finite]
        if path.shape[0] == 0:
            continue

        colour = _CAR_COLOURS[index % len(_CAR_COLOURS)]
        label = labels.get(agent, agent)
        (line,) = axes.plot(
            path[:, 0],
            path[:, 1],
            color=colour,
            linewidth=2.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=label,
            zorder=4,
        )
        handles.append(line)
        axes.scatter(
            path[:, 0],
            path[:, 1],
            color=colour,
            s=7,
            alpha=0.25,
            linewidths=0,
            zorder=3,
        )
        axes.scatter(
            path[0, 0],
            path[0, 1],
            color=colour,
            edgecolors=_CHART_SURFACE,
            marker="o",
            s=70,
            linewidths=1.4,
            zorder=5,
        )
        axes.scatter(
            path[-1, 0],
            path[-1, 1],
            color=colour,
            edgecolors=_CHART_SURFACE,
            marker="X",
            s=90,
            linewidths=1.4,
            zorder=6,
        )

    axes.set_title(title, color=_PRIMARY_INK, fontsize=11)
    _style_axes(axes)
    if handles:
        legend = axes.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)
        legend.get_frame().set_edgecolor(_GRIDLINE)

    figure.savefig(output_path, dpi=160, format="png", facecolor=_CHART_SURFACE)
    figure.clear()
    return output_path


def write_trajectory_sample(
    output_path: Path,
    *,
    trajectory: list[tuple[float, float]],
    completed: bool,
    dnf_reason: str | None,
    trial_id: str,
    seed: int,
) -> Path:
    """Persist one time-trial's raw XY samples for later per-checkpoint aggregation.

    Individual per-trial PNGs don't scale (N trials -> N images); instead each
    trial's path is stashed here as compact data, and
    `write_checkpoint_trajectory_plot` rebuilds the single per-checkpoint PNG
    from every stashed trial whenever a new one lands - including trials
    written by an earlier, resumed process.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xy = np.asarray(trajectory, dtype=np.float64).reshape(-1, 2)
    np.savez_compressed(
        output_path,
        xy=xy,
        completed=np.asarray(completed),
        dnf_reason=np.asarray(dnf_reason or ""),
        trial_id=np.asarray(trial_id),
        seed=np.asarray(seed),
    )
    return output_path


def _load_trajectory_samples(directory: Path) -> list[dict]:
    samples = []
    if not directory.is_dir():
        return samples
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            samples.append(
                {
                    "xy": data["xy"],
                    "completed": bool(data["completed"]),
                    "dnf_reason": str(data["dnf_reason"]) or None,
                    "trial_id": str(data["trial_id"]),
                    "seed": int(data["seed"]),
                }
            )
    return samples


def write_checkpoint_trajectory_plot(
    output_path: Path,
    *,
    track_waypoints,
    trajectory_sample_directory: Path,
    checkpoint_id: str,
    algorithm: str,
    track_name: str | None = None,
) -> Path | None:
    """Render every stashed time trial for one checkpoint into a single PNG.

    Replaces the old one-PNG-per-trial output: for N trials this writes (or
    overwrites, as more trials land) exactly one file, so a 6-checkpoint
    campaign produces 6 trajectory images no matter how many trials each ran.
    Returns None if no trials have been stashed yet.
    """
    samples = _load_trajectory_samples(Path(trajectory_sample_directory))
    if not samples:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(10.0, 8.0), constrained_layout=True, facecolor=_CHART_SURFACE)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)

    handles = _draw_track(axes, track_waypoints, track_name)

    completed_samples = [sample for sample in samples if sample["completed"]]
    dnf_samples = [sample for sample in samples if not sample["completed"]]

    # More overlaid trials -> lower per-line alpha, so density (not the last
    # line drawn) is what reads as "the common racing line".
    def _line_alpha(count: int) -> float:
        return float(np.clip(6.0 / max(count, 1), 0.12, 0.8))

    completed_alpha = _line_alpha(len(completed_samples))
    dnf_alpha = _line_alpha(len(dnf_samples))

    for sample in completed_samples:
        path = sample["xy"]
        finite = np.isfinite(path).all(axis=1)
        path = path[finite]
        if path.shape[0] == 0:
            continue
        axes.plot(
            path[:, 0],
            path[:, 1],
            color=_STATUS_GOOD,
            linewidth=1.6,
            solid_capstyle="round",
            alpha=completed_alpha,
            zorder=3,
        )

    for sample in dnf_samples:
        path = sample["xy"]
        finite = np.isfinite(path).all(axis=1)
        path = path[finite]
        if path.shape[0] == 0:
            continue
        axes.plot(
            path[:, 0],
            path[:, 1],
            color=_STATUS_CRITICAL,
            linewidth=1.6,
            linestyle=(0, (4, 2)),
            solid_capstyle="round",
            alpha=dnf_alpha,
            zorder=3,
        )
        axes.scatter(
            path[-1, 0],
            path[-1, 1],
            color=_STATUS_CRITICAL,
            edgecolors=_CHART_SURFACE,
            marker="X",
            s=60,
            linewidths=1.2,
            alpha=min(1.0, dnf_alpha + 0.35),
            zorder=4,
        )

    total = len(samples)
    completion_rate = len(completed_samples) / total if total else 0.0
    handles.append(
        Line2D(
            [0], [0], color=_STATUS_GOOD, linewidth=2.0,
            label=f"completed ({len(completed_samples)})",
        )
    )
    handles.append(
        Line2D(
            [0], [0], color=_STATUS_CRITICAL, linewidth=2.0, linestyle=(0, (4, 2)),
            label=f"DNF ({len(dnf_samples)})",
        )
    )

    axes.set_title(
        f"{checkpoint_id} ({algorithm}) - time trials: "
        f"{len(completed_samples)}/{total} completed ({completion_rate:.0%})",
        color=_PRIMARY_INK,
        fontsize=11,
    )
    _style_axes(axes)
    legend = axes.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)
    legend.get_frame().set_edgecolor(_GRIDLINE)

    figure.savefig(output_path, dpi=180, format="png", facecolor=_CHART_SURFACE)
    figure.clear()
    return output_path
