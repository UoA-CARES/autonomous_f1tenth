from rclpy import Future
from std_msgs.msg import Empty
from .controller import Controller
from environments.util import reduce_lidar, process_odom

class RLController(Controller):

    def __init__(self, node_name, car_name, step_length, observation_mode='lidar_only'):
        super().__init__(node_name, car_name, step_length)
        
        self.observation_mode = observation_mode
        
        match(observation_mode):
            case 'lidar_only':
                self.OBSERVATION_SIZE = 12
            case 'no_position':
                self.OBSERVATION_SIZE = 16
            case 'full':
                self.OBSERVATION_SIZE = 18
            case _:
                raise ValueError(f'Invalid observation mode: {observation_mode}')
        
        self.ACTION_NUM = 2
        

    def get_observation(self):
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        
        if self.observation_mode == 'lidar_only':
            lidar = reduce_lidar(lidar)
            odom = odom[-2:]

        return odom + lidar

    def step(self, action):

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        self.sleep()

        self.timer_future = Future()

        state = self.get_observation()
        self.update_goal_position(state[0:2], state[-2:])

        return state