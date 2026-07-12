import numpy as np

from .goal_positions import goal_positions
from .track_progress_model import TrackProgressModel
from .waypoints import waypoints


def get_all_goals_and_waypoints_in_multi_tracks(track_name):
    all_car_goals = {}
    all_car_waypoints = {}
    if track_name == "multi_track_full":
        austin_gp = goal_positions["austin_track"]
        budapest_gp = [[x + 200, y] for x, y in goal_positions["budapest_track"]]
        hockenheim_gp = [[x + 300, y] for x, y in goal_positions["hockenheim_track"]]
        melbourne_gp = [[x + 500, y] for x, y in goal_positions["melbourne_track"]]
        saopaolo_gp = [[x + 600, y] for x, y in goal_positions["saopaolo_track"]]
        shanghai_gp = [[x + 750, y] for x, y in goal_positions["shanghai_track"]]
        all_car_goals = {
            "austin_track": austin_gp,
            "budapest_track": budapest_gp,
            "hockenheim_track": hockenheim_gp,
            "melbourne_track": melbourne_gp,
            "saopaolo_track": saopaolo_gp,
            "shanghai_track": shanghai_gp,
        }
        austin_wp = waypoints["austin_track"]
        budapest_wp = [
            (x + 200, y, yaw, index) for x, y, yaw, index in waypoints["budapest_track"]
        ]
        hockenheim_wp = [
            (x + 300, y, yaw, index)
            for x, y, yaw, index in waypoints["hockenheim_track"]
        ]
        melbourne_wp = [
            (x + 500, y, yaw, index)
            for x, y, yaw, index in waypoints["melbourne_track"]
        ]
        saopaolo_wp = [
            (x + 600, y, yaw, index) for x, y, yaw, index in waypoints["saopaolo_track"]
        ]
        shanghai_wp = [
            (x + 750, y, yaw, index) for x, y, yaw, index in waypoints["shanghai_track"]
        ]
        all_car_waypoints = {
            "austin_track": austin_wp,
            "budapest_track": budapest_wp,
            "hockenheim_track": hockenheim_wp,
            "melbourne_track": melbourne_wp,
            "saopaolo_track": saopaolo_wp,
            "shanghai_track": shanghai_wp,
        }
    elif track_name == "multi_track":
        austin_gp = goal_positions["austin_track"]
        budapest_gp = [[x + 200, y] for x, y in goal_positions["budapest_track"]]
        hockenheim_gp = [[x + 300, y] for x, y in goal_positions["hockenheim_track"]]
        all_car_goals = {
            "austin_track": austin_gp,
            "budapest_track": budapest_gp,
            "hockenheim_track": hockenheim_gp,
        }
        austin_wp = waypoints["austin_track"]
        budapest_wp = [
            (x + 200, y, yaw, index) for x, y, yaw, index in waypoints["budapest_track"]
        ]
        hockenheim_wp = [
            (x + 300, y, yaw, index)
            for x, y, yaw, index in waypoints["hockenheim_track"]
        ]
        all_car_waypoints = {
            "austin_track": austin_wp,
            "budapest_track": budapest_wp,
            "hockenheim_track": hockenheim_wp,
        }
    elif track_name == "multi_track_testing":
        melbourne_gp = goal_positions["melbourne_track"]
        saopaolo_gp = [[x + 100, y] for x, y in goal_positions["saopaolo_track"]]
        shanghai_gp = [[x + 250, y] for x, y in goal_positions["shanghai_track"]]
        all_car_goals = {
            "melbourne_track": melbourne_gp,
            "saopaolo_track": saopaolo_gp,
            "shanghai_track": shanghai_gp,
        }
        melbourne_wp = waypoints["melbourne_track"]
        saopaolo_wp = [
            (x + 100, y, yaw, index) for x, y, yaw, index in waypoints["saopaolo_track"]
        ]
        shanghai_wp = [
            (x + 250, y, yaw, index) for x, y, yaw, index in waypoints["shanghai_track"]
        ]
        all_car_waypoints = {
            "melbourne_track": melbourne_wp,
            "saopaolo_track": saopaolo_wp,
            "shanghai_track": shanghai_wp,
        }
    elif track_name == "multi_track_wide":
        all_car_goals = None
        track_01_wp = waypoints["track_01"]
        track_02_wp = [
            (x + 30, y, yaw, index) for x, y, yaw, index in waypoints["track_02"]
        ]
        track_03_wp = [
            (x + 60, y, yaw, index) for x, y, yaw, index in waypoints["track_03"]
        ]
        track_04_wp = [
            (x + 90, y, yaw, index) for x, y, yaw, index in waypoints["track_04"]
        ]
        track_05_wp = [
            (x + 120, y, yaw, index) for x, y, yaw, index in waypoints["track_05"]
        ]
        track_06_wp = [
            (x + 150, y, yaw, index) for x, y, yaw, index in waypoints["track_06"]
        ]
        all_car_waypoints = {
            "track_01": track_01_wp,
            "track_02": track_02_wp,
            "track_03": track_03_wp,
            "track_04": track_04_wp,
            "track_05": track_05_wp,
            "track_06": track_06_wp,
        }
    elif track_name == "multi_track_01":
        track_offsets = [
            ("track_01_150", 0),
            ("track_01_200", 30),
            ("track_01_250", 60),
            ("track_01_300", 90),
            ("track_01_350", 120),
            ("track_02_150", 150),
            ("track_02_200", 180),
            ("track_02_250", 210),
            ("track_02_300", 240),
            ("track_02_350", 270),
            ("track_03_150", 300),
            ("track_03_200", 330),
            ("track_03_250", 360),
            ("track_03_300", 390),
            ("track_03_350", 420),
            ("track_04_150", 450),
            ("track_04_200", 480),
            ("track_04_300", 540),
            ("track_04_350", 570),
            ("track_04_250", 660),
            ("track_05_150", 750),
            ("track_05_200", 780),
            ("track_05_250", 810),
            ("track_05_300", 840),
            ("track_05_350", 870),
            ("track_06_150", 900),
            ("track_06_200", 930),
            ("track_06_250", 960),
            ("track_06_300", 990),
            ("track_06_350", 1020),
        ]
        all_car_goals = None
        all_car_waypoints = {}
        for resolved_track_name, x_offset in track_offsets:
            source_waypoints = waypoints[resolved_track_name]
            global_wp = [
                (x + x_offset, y, yaw, index)
                for x, y, yaw, index in source_waypoints
            ]
            all_car_waypoints.update({resolved_track_name: global_wp})
    elif track_name == "multi_track_02":
        widths = [350]
        tracks = [
            "track_01",
            "track_02",
            "track_03",
            "track_04",
            "track_05",
            "track_06",
        ]
        all_car_goals = None
        all_car_waypoints = {}
        i = 0
        for track in tracks:
            for width in widths:
                resolved_track_name = f"{track}_{str(width)}"
                global_wp = [
                    (x + i * 30, y, yaw, index) for x, y, yaw, index in waypoints[track]
                ]
                all_car_waypoints.update({resolved_track_name: global_wp})
                i += 1
    elif track_name == "multi_track_test_01":
        widths = [150, 200, 250, 300, 350]
        tracks = ["test_track_01", "test_track_02"]
        all_car_goals = None
        all_car_waypoints = {}
        i = 0
        for track in tracks:
            for width in widths:
                resolved_track_name = f"{track}_{str(width)}"
                global_wp = [
                    (x + i * 30, y, yaw, index) for x, y, yaw, index in waypoints[track]
                ]
                all_car_waypoints.update({resolved_track_name: global_wp})
                i += 1
    elif track_name == "staged_tracks":
        widths = [350, 300, 250, 200, 150]
        tracks = [
            "track_01",
            "track_02",
            "track_03",
            "track_04",
            "track_05",
            "track_06",
        ]
        all_car_goals = None
        all_car_waypoints = {}
        i = 0
        for width in widths:
            for track in tracks:
                resolved_track_name = f"{track}_{str(width)}"
                global_wp = [
                    (x + i * 30, y, yaw, index)
                    for x, y, yaw, index in waypoints[resolved_track_name]
                ]
                all_car_waypoints.update({resolved_track_name: global_wp})
                i += 1
    elif track_name == "narrow_multi_track":
        all_car_goals = None
        vary_track_width_new_wp = [
            (x + 1, y, yaw, index)
            for x, y, yaw, index in waypoints["vary_track_width_new"]
        ]
        spiral_track_wp = [
            (x + 22, y, yaw, index) for x, y, yaw, index in waypoints["spiral_track"]
        ]
        track_01_1m_wp = [
            (x + 40, y, yaw, index) for x, y, yaw, index in waypoints["track_01_1m"]
        ]
        track_02_1m_wp = [
            (x + 49, y, yaw, index) for x, y, yaw, index in waypoints["track_02_1m"]
        ]
        track_03_1m_wp = [
            (x + 58, y, yaw, index) for x, y, yaw, index in waypoints["track_03_1m"]
        ]
        track_04_1m_wp = [
            (x + 67, y, yaw, index) for x, y, yaw, index in waypoints["track_04_1m"]
        ]
        track_05_1m_wp = [
            (x + 76, y, yaw, index) for x, y, yaw, index in waypoints["track_05_1m"]
        ]
        track_06_1m_wp = [
            (x + 85, y, yaw, index) for x, y, yaw, index in waypoints["track_06_1m"]
        ]
        narrow_track_01_wp = [
            (x + 94, y, yaw, index) for x, y, yaw, index in waypoints["narrow_track_01"]
        ]
        narrow_track_02_wp = [
            (x + 125, y, yaw, index)
            for x, y, yaw, index in waypoints["narrow_track_02"]
        ]
        narrow_track_03_wp = [
            (x + 156, y, yaw, index)
            for x, y, yaw, index in waypoints["narrow_track_03"]
        ]
        narrow_track_04_wp = [
            (x + 187, y, yaw, index)
            for x, y, yaw, index in waypoints["narrow_track_04"]
        ]
        narrow_track_05_wp = [
            (x + 218, y, yaw, index)
            for x, y, yaw, index in waypoints["narrow_track_05"]
        ]
        narrow_track_06_wp = [
            (x + 249, y, yaw, index)
            for x, y, yaw, index in waypoints["narrow_track_06"]
        ]
        track_01_2m_wp = [
            (x + 265, y, yaw, index) for x, y, yaw, index in waypoints["track_01_2m"]
        ]
        track_02_2m_wp = [
            (x + 281, y, yaw, index) for x, y, yaw, index in waypoints["track_02_2m"]
        ]
        track_03_2m_wp = [
            (x + 297, y, yaw, index) for x, y, yaw, index in waypoints["track_03_2m"]
        ]
        track_04_2m_wp = [
            (x + 313, y, yaw, index) for x, y, yaw, index in waypoints["track_04_2m"]
        ]
        track_05_2m_wp = [
            (x + 329, y, yaw, index) for x, y, yaw, index in waypoints["track_05_2m"]
        ]
        track_06_2m_wp = [
            (x + 345, y, yaw, index) for x, y, yaw, index in waypoints["track_06_2m"]
        ]
        all_car_waypoints = {
            "vary_track_width_new": vary_track_width_new_wp,
            "spiral_track": spiral_track_wp,
            "track_01_1m": track_01_1m_wp,
            "track_02_1m": track_02_1m_wp,
            "track_03_1m": track_03_1m_wp,
            "track_04_1m": track_04_1m_wp,
            "track_05_1m": track_05_1m_wp,
            "track_06_1m": track_06_1m_wp,
            "narrow_track_01": narrow_track_01_wp,
            "narrow_track_02": narrow_track_02_wp,
            "narrow_track_03": narrow_track_03_wp,
            "narrow_track_04": narrow_track_04_wp,
            "narrow_track_05": narrow_track_05_wp,
            "narrow_track_06": narrow_track_06_wp,
            "track_01_2m": track_01_2m_wp,
            "track_02_2m": track_02_2m_wp,
            "track_03_2m": track_03_2m_wp,
            "track_04_2m": track_04_2m_wp,
            "track_05_2m": track_05_2m_wp,
            "track_06_2m": track_06_2m_wp,
        }
    return all_car_goals, all_car_waypoints


def get_track_progress_models(tracks_waypoints: dict) -> dict[str, TrackProgressModel]:
    track_progress_models = {}
    for track_name in tracks_waypoints.keys():
        track_progress_models[track_name] = TrackProgressModel(
            np.array(tracks_waypoints[track_name])[:, :2]
        )
    return track_progress_models
