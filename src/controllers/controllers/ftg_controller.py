from rclpy import Future
from std_msgs.msg import Empty
from environments.track_reset import track_info
from environments.termination import reached_goal
from environments.util import reduce_lidar, process_odom
from .controller import Controller

class FTGController(Controller):

    def __init__(self, node_name, car_name, step_length, track_name):
        super().__init__(node_name, car_name, step_length)

        self.reset_sub = self.create_subscription(Empty, f'/reset', self.reset_cb, 10)

        self.goals = track_info[track_name]['goals']
        self.reset = self.goals[0]

        self.goal_position = self.goals[1]

    def reset_cb(self, msg):
        self.get_logger().info('Reset Detected from FTG Controller')
        self.goal_position = self.goals[1]
    
    def get_observation(self):
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        lidar = reduce_lidar(lidar)

        return odom + lidar + self.goal_position

    def step(self, action):

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        self.sleep()

        self.timer_future = Future()

        state = self.get_observation()
        self.update_goal_position(state[0:2], state[-2:])

        return state

    def update_goal_position(self, car_position, goal_position):
        
        if reached_goal(car_position, goal_position, 3):
            self.goal_position = self.goals[self.goals.index(goal_position) + 1]
