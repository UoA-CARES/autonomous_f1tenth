from collections import namedtuple
import numpy as np
import numpy.typing as npt
from scipy.interpolate import splrep, splev
from scipy import integrate, optimize, interpolate
from matplotlib import pyplot as plt
import warnings

# austin_coord = np.array(waypoints['budapest_track'])[:,:2]
# print(austin_coord.shape)
# austin_coord = np.append(austin_coord, [austin_coord[0]],axis=0)
# print(austin_coord.shape)

class TrackMathDef():
    '''Give mathematical definition of a looping track & provide supporting functions.'''
    
    def __init__(self, points:list):
        '''Need list of [x,y] coord to contruct a b-spline representation of track. Will connect the first point and the last point in array.'''
        points = np.array(points)
        points = np.append(points, [points[0]], axis=0)
        x_points = points[:, 0]
        y_points = points[:, 1]

        t = np.linspace(0, 1, len(x_points))

        self.para_x = interpolate.splrep(t, x_points)
        self.para_y = interpolate.splrep(t, y_points)
    
    def get_spline_point(self, t:float) -> npt.NDArray:
        '''Get [x,y] given t.'''
        x = float(interpolate.splev(t,self.para_x))
        y = float(interpolate.splev(t,self.para_y))
        return np.array([x,y])
    
    def distance_to_spline_minimize_target(self, t, coord:npt.NDArray):
        spline_coord = self.get_spline_point(t)
        return np.linalg.norm(spline_coord - coord)
    
    def get_closest_point_on_spline(self, coord:npt.NDArray, t_only = False):
        '''Get closest point's t value. If t_only false: give (t,[x,y])'''
        result = optimize.differential_evolution(self.distance_to_spline_minimize_target, popsize=25, maxiter=1500, tol=0.001, bounds=[(0, 1)], args=([coord]))
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
        '''IMPORTANT: assume t1 to t2 follows the direction of the path'''
        # warnings.filterwarnings('error')
        # try:
        length, _= integrate.quad(self.arc_length_integrand, t1, t2)
        # except integrate.IntegrationWarning:
        #     print(f" ERROR INTEGRATING: t1={t1}, t2={t2}")
        #     length = abs( np.linalg.norm(self.get_spline_point(t1) - self.get_spline_point(t2)))
        #     raise integrate.IntegrationWarning
        return length
    
    
    def get_distance_along_track_parametric(self, t1, t2):
        if t1 <= t2:
            if t2 - t1 < 0.5:
                # going forward normally
                return self.get_arc_length_forward(t1, t2)

            else:
                # going backward passing origin
                print("SPECIALI CASE: BACKING PAST ORIGIN")
                return - self.get_arc_length_forward(0,t1) - self.get_arc_length_forward(t2,1)

        else: # t1 > t2
            if t1 - t2 < 0.5:
                # going backward
                return -self.get_arc_length_forward(t2,t1)
            else:
                # going forward passing origin
                print("SPECIALI CASE: FORWARD PAST ORIGIN")
                return self.get_arc_length_forward(t1,1) + self.get_arc_length_forward(0,t2)

    def get_distance_along_track(self, coord_1, coord_2):
        t1 = self.get_closest_point_on_spline(coord_1, t_only=True)
        t2 = self.get_closest_point_on_spline(coord_2, t_only=True)

        return self.get_distance_along_track_parametric(t1, t2)

    def arc_length_minimize_target(self, t2, t1, dist):
        return self.get_distance_along_track_parametric(t1,t2) - dist
    
    def get_coord_on_track_from_distance(self, t_orig, dist):
    
        result = optimize.root(self.arc_length_minimize_target, t_orig, args=(t_orig, dist))
        
        if result.success:
            t2 = result.x[0] % 1
            return self.get_spline_point(t2)
        else:
            raise Exception("Getting target coord on track: did not converge.")


# goals, waypoints = get_all_goals_and_waypoints_in_multi_tracks("multi_track")
# print("getting parametric")

# coords = np.array(waypoints['budapest_track'])[:,:2]

# track_def = TrackMathDef(coords)

# p1 = [204, -4]
# p2 = [202,0]
# p3 = [198,0]


# xx = np.linspace(0,1,1000)

# tx, cx, kx = track_def.para_x
# ty, cy, ky = track_def.para_y

# p4 = track_def.get_coord_on_track_from_distance(0.01, -20)

# spline_x = interpolate.BSpline(tx, cx, kx, False)
# spline_y = interpolate.BSpline(ty, cy, ky, False)

# print(track_def.get_distance_along_track(p1,p2))
# _,mapped_coord = track_def.get_closest_point_on_spline(p1,t_only=False)
# print(mapped_coord)
# _,mapped_coord2 = track_def.get_closest_point_on_spline(p2,t_only=False)
# print(mapped_coord2)


# print("plotting")
# plt.plot(spline_x(xx),spline_y(xx), 'r', label='BSpline')
# plt.plot(mapped_coord[0], mapped_coord[1],'go')
# plt.plot(p1[0],p1[1],'gx')
# plt.plot(mapped_coord2[0], mapped_coord2[1],'ro')
# plt.plot(p2[0],p2[1],'rx')

# plt.plot(p4[0],p4[1],'bx')

# plt.grid()
# plt.legend(loc='best')
# plt.show()