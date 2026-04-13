import math
from typing import Sequence

from scipy.spatial.transform import Rotation


def get_quaternion_from_euler(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert roll/pitch/yaw Euler angles to quaternion in [x, y, z, w] order."""
    qx, qy, qz, qw = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return [float(qx), float(qy), float(qz), float(qw)]


def get_euler_from_quaternion(
    w: float, x: float, y: float, z: float
) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw)."""
    # scipy expects xyzw order
    roll, pitch, yaw = Rotation.from_quat([x, y, z, w]).as_euler("xyz")
    return roll, pitch, yaw


def twist_to_ackermann(omega: float, linear_v: float, wheelbase_m: float) -> float:
    """Convert angular velocity to steering angle using Ackermann geometry."""
    if math.isclose(linear_v, 0.0):
        return 0.0
    if math.isclose(wheelbase_m, 0.0):
        return 0.0
    delta = math.atan((wheelbase_m * omega) / linear_v)
    return delta


def ackermann_to_twist(delta: float, linear_v: float, wheelbase_m: float) -> float:
    """Convert steering angle to angular velocity using Ackermann geometry."""
    if math.isclose(wheelbase_m, 0.0):
        return 0.0
    omega = math.tan(delta) * linear_v / wheelbase_m
    return omega


def has_flipped_over(
    quaternion_wxyz: Sequence[float], tilt_limit_rad: float = math.radians(60.0)
) -> bool:
    """
    Returns True if roll or pitch exceeds tilt_limit_rad.
    Expects quaternion in [w, x, y, z] order.
    """
    _, x, y, _ = quaternion_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)  # z-component of rotated world up vector
    return up_z < math.cos(tilt_limit_rad)


def lateral_translation(
    spline_location: tuple[float, float], angle: float, shift: float
) -> tuple[float, float]:
    """Translate a point laterally by `shift` meters relative to heading `angle`."""
    x, y = spline_location
    x1 = x + shift * math.cos(angle + (math.pi / 2))
    y1 = y + shift * math.sin(angle + (math.pi / 2))
    return x1, y1
