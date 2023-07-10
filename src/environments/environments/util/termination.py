

def has_collided(lidar_ranges, collision_range):
    return any(0 < ray < collision_range for ray in lidar_ranges)

def has_flipped_over(quaternion):
    _, x, y, _ = quaternion
    return abs(x) > 0.5 or abs(y) > 0.5