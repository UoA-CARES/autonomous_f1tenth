import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev
from scipy import integrate, optimize, interpolate

class TrackMathDef():
    
    def __init__(self, points:list):
        points = np.array(points)
        points = np.append(points, [points[0]], axis=0)
        x_points = points[:, 0]
        y_points = points[:, 1]
        t = np.linspace(0, 1, len(x_points))
        self.para_x = interpolate.splrep(t, x_points, k=2)
        self.para_y = interpolate.splrep(t, y_points, k=2)
    
    def get_spline_point(self, t:float) -> npt.NDArray:
        x = float(interpolate.splev(t,self.para_x))
        y = float(interpolate.splev(t,self.para_y))
        return np.array([x,y])
    
    def get_distance_to_spline_point(self, t, coord:npt.NDArray):
        spline_coord = self.get_spline_point(t)
        return np.linalg.norm(spline_coord - coord)
    
    def get_closest_point_on_spline(self, coord:npt.NDArray, t_only = False):
        result = optimize.differential_evolution(self.get_distance_to_spline_point, bounds=[(0, 1)], args=([coord]))
        t_optimal = result.x[0]

        if t_only:
            return t_optimal
        else:
            return t_optimal, self.get_spline_point(t_optimal)
    
    def arc_length_integrand(self, t):
        dxdt = splev(t % 1, self.para_x, der=1)
        dydt = splev(t % 1, self.para_y, der=1)
        return np.sqrt(dxdt**2 + dydt**2)

    def get_arc_length_forward(self, t1, t2):
        # IMPORTANT: assume t1 to t2 follows the direction of the path
        length, _= integrate.quad(self.arc_length_integrand, t1, t2)
        return length
    
    
    def get_distance_along_track_parametric(self, t1, t2, approximate=False):
        ''' Calculate arc length along track. If approximate is true, simply take the absolute distance between the two points with the rationale being
        given the small amount of time per step the progress can likely be approximated with a straight line.'''
        if approximate:
            return np.linalg.norm(self.get_spline_point(t1) - self.get_spline_point(t2))
        if t1 <= t2:
            if t2 - t1 < 0.5:
                # going forward normally
                return self.get_arc_length_forward(t1, t2)
            else:
                # going backward passing origin
                print("SPECIAL CASE: BACKING PAST ORIGIN")
                return - self.get_arc_length_forward(0,t1) - self.get_arc_length_forward(t2,1)
        else: 
            if t1 - t2 < 0.5:
                # going backward
                return -self.get_arc_length_forward(t2,t1)
            else:
                # going forward passing origin
                print("SPECIAL CASE: FORWARD PAST ORIGIN")
                return self.get_arc_length_forward(t1,1) + self.get_arc_length_forward(0,t2)

    def get_distance_along_track(self, coord_1, coord_2):
        t1 = self.get_closest_point_on_spline(coord_1, t_only=True)
        t2 = self.get_closest_point_on_spline(coord_2, t_only=True)
        return self.get_distance_along_track_parametric(t1, t2, approximate=True)

    def arc_length_minimize_target(self, t2, t1, dist):
        return self.get_distance_along_track_parametric(t1,t2) - dist
    
    def get_coord_on_track_from_distance(self, t_orig, dist):
        result = optimize.root(self.arc_length_minimize_target, t_orig, args=(t_orig, dist))
        if result.success:
            t2 = result.x[0] % 1
            return self.get_spline_point(t2)
        else:
            raise Exception("Getting target coord on track: did not converge.")