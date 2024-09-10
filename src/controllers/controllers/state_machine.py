import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from message_filters import Subscriber
from nav_msgs.msg import Odometry
from .util import absoluteDistance
import numpy as np

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        self.state_pub = self.create_publisher(
            String,
            '/state',
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/f1tenth/odometry',
            self.odom_callback,
            10
        )   
    
        self.currState = 'setup'
        self.odom = []
        self.init_odom = []

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.odom = [position.x, position.y]
        #self.get_logger().info("reading odom")
    
    def pubState(self, str):
        msg = String()
        msg.data = str
        self.state_pub.publish(msg)

    def getCurrState(self):
        return self.currState
    
    def changeState(self, newState):
        self.currState = newState

def main():
    rclpy.init()
    state_machine = StateMachine()
    print("In state machine")
    # Get initial odometry
    while(len(state_machine.odom) == 0):
        rclpy.spin_once(state_machine)
        state_machine.get_logger().info(str(state_machine.odom))
        time.sleep(2)
    state_machine.init_odom = state_machine.odom

    #Waiting to move
    while (absoluteDistance(np.array(state_machine.init_odom), np.array(state_machine.odom)) < 0.2):
        rclpy.spin_once(state_machine)
        time.sleep(0.1)
    state_machine.changeState("mapping")


    state_machine.destroy_node()
    rclpy.shutdown()
        


if __name__ == '__main__':
    main()