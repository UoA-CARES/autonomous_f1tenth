from collections import namedtuple

Waypoint = namedtuple('Waypoint', ['x', 'y', 'Y', 'index'])


waypoints = {
    'track_1': [
        Waypoint(-14.11, 3.71, -1.81),
        Waypoint(-15.75, -1.71, -1.96),
        Waypoint(-16.71, -5.48, -1.35),
        Waypoint(-7.53, -5.43, -0.27),
        Waypoint(-0.22, -8.55, -0.87),
        Waypoint(1.9, -12.05, -0.66),
        Waypoint(5.5, -12.38, 0.79),
        Waypoint(7.32, -7.3, 0.97),
        Waypoint(9.62, -1.45, 1.23),
        Waypoint(10.76, 4.08, 1.64),
        Waypoint(8.48, 9.77, 2.07),
        Waypoint(4.93, 13.27, -3.06),
        Waypoint(1.5, 9.96, -1.9),
        Waypoint(-0.89, 5.67, -2.94),
        Waypoint(-5.52, 10.83, 2.02),
        Waypoint(-6.67, 15.67, 2.38),
        Waypoint(-10.84, 13.99, -1.93),
        Waypoint(12.81, 8.14, -1.93),
    ],
    'austin_track': [
        Waypoint(0, 0, -0.64, 0),
        Waypoint(7.29, -5.57, -0.64, 8),
        Waypoint(20.05, -15.31, -0.64, 22),
        Waypoint(30.38, -23.2, -0.64, 34),
        Waypoint(37.88, -28.45, -0.39, 41),
        Waypoint(40.65, -26.31, 1.94, 43),
        Waypoint(40.24, -7.39, 0.66, 64),
        Waypoint(70.28, 22.93, 0.08, 104),
        Waypoint(106.15, 25.015, 0.08, 140),
        Waypoint(124.86, 41.54, 0.83, 162),
        Waypoint(130.48, 48.79, 0.83, 171),
        Waypoint(117.07, 47.39, -2.87, 187),
        Waypoint(80, 38.14, -2.87, 219),
        Waypoint(44.39, 33.27, -3.04, 251),
        Waypoint(44.23, 28.37, -0.85, 259),
        Waypoint(38.34, 15.79, -1.07, 299),
        Waypoint(14.71, 14.25, 2.3, 335),
    ],
    'budapest_track': [
        Waypoint(0, 0, 2.45, 0),
        Waypoint(-11.7, 9.66, 2.45, 12),
        Waypoint(-26.62, 21.92, 2.45, 26),
        Waypoint(-43.45, 36.17, 2.45, 42),
        Waypoint(0.6, 53.88, 1, 103),
        Waypoint(29.87, 88.89, -0.77, 154),
        Waypoint(49.35, 38.16, -1.01, 200),
        Waypoint(12.95, 11.73, -2.94, 253),
    ],
    'hockenheim_track': [
        Waypoint(0, 0, 1.99, 0),
        Waypoint(-7.37, 14.8, 1.99, 15),
        Waypoint(26.5, 40.58, -0.53, 70),
        Waypoint(63.19, 29.06, -0.02, 103),
        Waypoint(93.67, 37.96, 0.36, 130),
        Waypoint(80.3, 23.3, -2.53, 138),
        Waypoint(39.86, -2.53, -2.53, 221),
        Waypoint(9.98, 8.54, -2.53, 254),
        Waypoint(4.92, -10.3, 2.03, 296),
    ],


    
}