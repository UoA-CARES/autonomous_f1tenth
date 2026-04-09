# Backward-compatible util module re-exporting all utilities.
# Split into focused submodules for maintainability.

from .geometry_utils import (
    ackermann_to_twist,
    find_occurrences,
    generate_position,
    get_euler_from_quarternion,
    get_quaternion_from_euler,
    has_flipped_over,
    lateral_translation,
    process_odom,
    twist_to_ackermann,
)
from .lidar_utils import (
    avg_lidar,
    avg_lidar_w_consensus,
    create_lidar_msg,
    forward_reduce_lidar,
    has_collided,
    process_ae_lidar,
    process_ae_lidar_beta_vae,
    process_lidar_med_filt,
    reconstruct_ae_latent,
    uneven_median_lidar,
)
from .track_utils import (
    get_all_goals_and_waypoints_in_multi_tracks,
    get_track_math_defs,
)

__all__ = [
    # Geometry
    "get_quaternion_from_euler",
    "get_euler_from_quarternion",
    "generate_position",
    "process_odom",
    "twist_to_ackermann",
    "ackermann_to_twist",
    "has_flipped_over",
    "lateral_translation",
    "find_occurrences",
    # Lidar
    "avg_lidar",
    "avg_lidar_w_consensus",
    "uneven_median_lidar",
    "process_lidar_med_filt",
    "process_ae_lidar",
    "process_ae_lidar_beta_vae",
    "reconstruct_ae_latent",
    "create_lidar_msg",
    "forward_reduce_lidar",
    "has_collided",
    # Track
    "get_all_goals_and_waypoints_in_multi_tracks",
    "get_track_math_defs",
]

# Deprecated: all functions are available via individual imports
# Use new focused module imports (geometry_utils, lidar_utils, track_utils) directly
