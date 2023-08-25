from collections import namedtuple

Waypoint = namedtuple('Waypoint', ['x', 'y', 'Y'])


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
    'budapest': [
        Waypoint(0, 0, -0.66),
        Waypoint(7.27, -5.88, -0.66),
        Waypoint(14.19, -7.85, 0.74),
        Waypoint(13.44, -0.49, 2.45),
        Waypoint(7.74, 4.32, 2.04),
        Waypoint(15.5, 4.11, -0.9),
        Waypoint(21.58, -0.55, 0.76),
        Waypoint(31.42, 9.76, 0.76),
        Waypoint(38.02, 17.33, 1.26),
        Waypoint(34.74, 26.22, 2.16),
        Waypoint(34.83, 39.9, 1.7),
        Waypoint(25.33, 46.92, 1.7),
        Waypoint(58.7, 58.7, 2.17),
        Waypoint(14.53, 68.36, 2.59),
        Waypoint(7.71, 64.15, -1.48),
        Waypoint(7.8, 50.07, -2.17),
        Waypoint(-3.65, 30.22, -1.95),
        Waypoint(-2.49, 22.02, -1.2),
        Waypoint(-1.34, 17.12, -2.05),
        Waypoint(-9.16, 18.16, 2.49),
        Waypoint(-19.28, 26.05, 2.82),
        Waypoint(-28.16, 27.8, 3.11),
        Waypoint(-29.03, 23.69, -0.79),
        Waypoint(-20.17, 16.62, -0.75),
        Waypoint(-7.26, 5.93, -0.75),
    ],
    'austin': [
        Waypoint(0, 0, -0.66),
        Waypoint(8.76, -6.8, -0.66),
        Waypoint(21.59, -16.75, -0.66),
        Waypoint(32.13, -24.94, -0.66),
        Waypoint(38.69, -28.53, -0.02),
        Waypoint(39.99, -24.59, 1.91),
        Waypoint(37.01, -13.32, 1.43),
        Waypoint(45.58, -4, 0.54),
        Waypoint(53.58, 2.01, 0.95),
        Waypoint(),
        
    ]
    
}