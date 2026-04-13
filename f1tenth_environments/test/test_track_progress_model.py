import math

import numpy as np
import pytest

from f1tenth_environments.track_progress_model import TrackProgressModel


def _closed_circle_points(num_points: int = 80) -> list[list[float]]:
    return [
        [math.cos(theta), math.sin(theta)]
        for theta in np.linspace(0.0, 2.0 * math.pi, num_points, endpoint=False)
    ]


def _circular_delta(a: float, b: float) -> float:
    """Smallest absolute difference between two values on a unit circle domain."""
    return min((a - b) % 1.0, (b - a) % 1.0)


@pytest.fixture
def circle_model() -> TrackProgressModel:
    return TrackProgressModel(_closed_circle_points())


def test_requires_at_least_two_points() -> None:
    """Reject degenerate track definitions.

    A closed spline needs at least two distinct input points to define any path.
    This protects against accidentally constructing a track model from malformed
    configuration data and then getting confusing spline failures later.
    """
    with pytest.raises(ValueError):
        TrackProgressModel([[0.0, 0.0]])


def test_normalize_spline_coord_wraps_to_unit_interval(
    circle_model: TrackProgressModel,
) -> None:
    """Normalize arbitrary spline coordinates into the canonical one-lap interval.

    The chosen values exercise both overflow and underflow cases: `1.25` should
    wrap forward to `0.25`, and `-0.10` should wrap backward to `0.90`. This
    guards the core modulo behavior used throughout the class.
    """
    assert circle_model._normalize_spline_coord(1.25) == pytest.approx(0.25)
    assert circle_model._normalize_spline_coord(-0.10) == pytest.approx(0.90)


def test_world_coord_projection_round_trip_for_exact_spline_points(
    circle_model: TrackProgressModel,
) -> None:
    """Round-trip exact on-curve points through world-to-spline projection.

    The selected spline coordinates sample several different parts of the lap,
    including early, mid, and late positions. Each generated world point should
    project back close to its original spline coordinate, proving that the
    projection routine is consistent with the forward spline evaluation.
    """
    for spline_coord in [0.05, 0.2, 0.37, 0.81]:
        world_xy = circle_model.spline_coord_to_world_coord_xy(spline_coord)
        projected = circle_model.world_coord_to_spline_coord(world_xy)
        assert _circular_delta(projected, spline_coord) < 0.02


def test_distance_from_spline_coord_to_world_coord_xy_is_zero_on_curve(
    circle_model: TrackProgressModel,
) -> None:
    """Distance-to-track helper should vanish for an exact point on the spline.

    The point is first generated from spline coordinate `0.33`, so querying the
    distance back to the track at that same coordinate should produce zero up to
    numerical tolerance. This protects the basic geometry contract of the helper.
    """
    spline_coord = 0.33
    world_xy = circle_model.spline_coord_to_world_coord_xy(spline_coord)
    distance = circle_model.distance_from_spline_coord_to_world_coord_xy(
        spline_coord,
        world_xy,
    )
    assert distance == pytest.approx(0.0, abs=1e-9)


def test_linear_distance_is_not_greater_than_unsigned_arc_distance(
    circle_model: TrackProgressModel,
) -> None:
    """Straight-line distance should never exceed path length along the track.

    The pair `0.13 -> 0.78` spans a visibly curved section of the circle, so the
    direct chord should be shorter than or equal to the corresponding unsigned
    arc distance. This guards against accidentally mixing Euclidean and arc
    metrics in the implementation.
    """
    from_coord = 0.13
    to_coord = 0.78
    linear = circle_model.linear_distance_between_spline_coords(from_coord, to_coord)
    arc_unsigned = circle_model.arc_distance_between_spline_coords(
        from_coord,
        to_coord,
        signed=False,
    )
    assert linear <= arc_unsigned + 1e-9


def test_linear_distance_between_world_coords_matches_spline_values_for_exact_points(
    circle_model: TrackProgressModel,
) -> None:
    """World-space distance API should agree with spline-space distance API.

    Both world points are produced directly from known spline coordinates, so the
    world/world path should reduce to the same computation as the spline/spline
    path after projection. This verifies that the projection layer is transparent
    for exact on-track inputs.
    """
    from_spline = 0.14
    to_spline = 0.41
    from_world = circle_model.spline_coord_to_world_coord_xy(from_spline)
    to_world = circle_model.spline_coord_to_world_coord_xy(to_spline)

    linear_spline = circle_model.linear_distance_between_spline_coords(
        from_spline,
        to_spline,
    )
    linear_world = circle_model.linear_distance_between_world_coords(
        from_world,
        to_world,
    )
    assert linear_world == pytest.approx(linear_spline, rel=1e-3, abs=1e-3)


def test_arc_length_integrand_matches_circle_circumference_density(
    circle_model: TrackProgressModel,
) -> None:
    """Arc-length integrand should match the expected density for a unit circle.

    On a unit circle parameterized over one full spline lap, the local rate of
    arc-length accumulation is approximately `2π` everywhere. Sampling at `0.25`
    checks that this helper reflects the intended geometric interpretation.
    """
    integrand = circle_model._arc_length_integrand(0.25)
    assert integrand == pytest.approx(2.0 * math.pi, rel=2e-2)


def test_signed_arc_distance_has_expected_wraparound_sign(
    circle_model: TrackProgressModel,
) -> None:
    """Wrap-around cases should preserve the intended forward/backward sign.

    Moving from `0.95` to `0.05` crosses the seam in the short forward direction,
    while `0.05` to `0.95` represents the corresponding short backward move.
    This guards the sign convention specifically at the lap boundary.
    """
    forward_wrap = circle_model.signed_arc_distance_between_spline_coords(0.95, 0.05)
    backward_wrap = circle_model.signed_arc_distance_between_spline_coords(0.05, 0.95)

    assert forward_wrap > 0.0
    assert backward_wrap < 0.0


def test_signed_arc_distance_is_antisymmetric_off_half_lap(
    circle_model: TrackProgressModel,
) -> None:
    """Signed arc distance should be antisymmetric away from tie cases.

    The chosen pair is intentionally not half a lap apart, so there is a unique
    shortest direction. Reversing the endpoints should preserve magnitude and
    flip only the sign, which is a core invariant of the signed distance API.
    """
    a, b = 0.11, 0.64
    dab = circle_model.signed_arc_distance_between_spline_coords(a, b)
    dba = circle_model.signed_arc_distance_between_spline_coords(b, a)
    assert dab == pytest.approx(-dba, rel=1e-6, abs=1e-6)


def test_half_lap_tie_prefers_forward_direction(
    circle_model: TrackProgressModel,
) -> None:
    """Exact half-lap ties should follow the class's documented forward convention.

    The pair `0.10 <-> 0.60` sits exactly half a lap apart on the normalized
    domain. In that ambiguous case the implementation currently chooses the
    forward/positive branch, and this test protects that deliberate convention.
    """
    forward_half_lap = circle_model.signed_arc_distance_between_spline_coords(
        0.10, 0.60
    )
    reverse_half_lap = circle_model.signed_arc_distance_between_spline_coords(
        0.60, 0.10
    )

    assert forward_half_lap > 0.0
    assert reverse_half_lap > 0.0
    assert forward_half_lap == pytest.approx(reverse_half_lap, rel=1e-5)


def test_world_coord_methods_match_spline_coord_methods(
    circle_model: TrackProgressModel,
) -> None:
    """Signed world-space distance should agree with signed spline-space distance.

    The world inputs are exact evaluations of known spline coordinates, so the
    extra projection step should not change the result beyond small numerical
    tolerance. This confirms that both public APIs represent the same geometry.
    """
    from_spline = 0.22
    to_spline = 0.47
    from_world = circle_model.spline_coord_to_world_coord_xy(from_spline)
    to_world = circle_model.spline_coord_to_world_coord_xy(to_spline)

    arc_spline = circle_model.signed_arc_distance_between_spline_coords(
        from_spline,
        to_spline,
    )
    arc_world = circle_model.signed_arc_distance_between_world_coords(
        from_world,
        to_world,
    )
    assert arc_world == pytest.approx(arc_spline, rel=1e-3, abs=1e-3)


def test_unwrapped_forward_arc_length_requires_non_decreasing_bounds(
    circle_model: TrackProgressModel,
) -> None:
    """Unwrapped forward integration must reject reversed intervals.

    The helper is specifically for monotonic unwrapped intervals, so passing
    `0.9 -> 0.2` should fail immediately rather than silently producing an
    invalid length. This protects a key assumption used by the inverse solver.
    """
    with pytest.raises(ValueError):
        circle_model._forward_arc_length_unwrapped(0.9, 0.2)


def test_forward_arc_length_wraps_across_seam(circle_model: TrackProgressModel) -> None:
    """Wrapped forward travel should match the equivalent unwrapped interval.

    The path from `0.95` to `0.05` crosses the seam once. Expressing the same
    motion as `0.95 -> 1.05` in unwrapped coordinates should yield the same arc
    length, which verifies the seam-handling logic in the forward helper.
    """
    wrapped = circle_model._forward_arc_length(0.95, 0.05)
    unwrapped = circle_model._forward_arc_length_unwrapped(0.95, 1.05)
    assert wrapped == pytest.approx(unwrapped, rel=1e-6, abs=1e-6)


def test_lap_length_is_cached_after_first_computation(
    circle_model: TrackProgressModel,
) -> None:
    """Lap length should be computed once and then served from cache.

    The first call populates the cache from an integration over the full spline.
    The second call should return the same value without changing it, protecting
    a simple but important performance optimization.
    """
    assert circle_model._lap_length_cache is None
    lap_first = circle_model._lap_length()
    assert circle_model._lap_length_cache == pytest.approx(lap_first)
    lap_second = circle_model._lap_length()
    assert lap_second == pytest.approx(lap_first)


def test_world_coord_xy_from_spline_distance_handles_multi_lap_inputs(
    circle_model: TrackProgressModel,
) -> None:
    """Positive inverse-distance queries should be stable across added full laps.

    The request `+0.42` and `+0.42 + 3 * lap_length` describe the same final XY
    location on a closed track. This ensures the inverse solver correctly reduces
    arbitrary positive distances modulo one lap.
    """
    origin = 0.2
    lap = circle_model._lap_length()

    d = 0.42
    base_xy = circle_model.world_coord_xy_from_spline_distance(origin, d)
    multi_lap_xy = circle_model.world_coord_xy_from_spline_distance(
        origin, d + 3.0 * lap
    )

    assert np.linalg.norm(base_xy - multi_lap_xy) < 1e-6


def test_world_coord_xy_from_spline_distance_handles_negative_multi_lap_inputs(
    circle_model: TrackProgressModel,
) -> None:
    """Negative inverse-distance queries should also be stable across full laps.

    The request `-0.35` and `-0.35 - 2 * lap_length` should end at the same XY
    point because extra full backward laps do not change final position on the
    closed loop. This protects the negative-distance branch of the solver.
    """
    origin = 0.61
    lap = circle_model._lap_length()

    d = -0.35
    base_xy = circle_model.world_coord_xy_from_spline_distance(origin, d)
    multi_lap_xy = circle_model.world_coord_xy_from_spline_distance(
        origin, d - 2.0 * lap
    )

    assert np.linalg.norm(base_xy - multi_lap_xy) < 1e-6


def test_world_coord_xy_from_spline_distance_zero_returns_origin_point(
    circle_model: TrackProgressModel,
) -> None:
    """Zero inverse distance should be an exact no-op.

    Asking for zero travel from spline coordinate `0.44` should return the same
    world point, with no unnecessary root-solving drift. This guards the early
    return path in the inverse-distance implementation.
    """
    origin = 0.44
    origin_xy = circle_model.spline_coord_to_world_coord_xy(origin)
    solved_xy = circle_model.world_coord_xy_from_spline_distance(origin, 0.0)
    assert np.linalg.norm(origin_xy - solved_xy) < 1e-9
