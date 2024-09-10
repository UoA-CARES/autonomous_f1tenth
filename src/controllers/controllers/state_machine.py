import rclpy
from rclpy.node import Node
from rclpy import Future
from std_msgs.msg import String
import time
from message_filters import Subscriber
from nav_msgs.msg import Odometry
from .util import absoluteDistance
import numpy as np
from .controller import Controller

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        
        self.init_odom = []
        self.odom = []

        self.state_pub = self.create_publisher(
            String,
            '/state',
            10
        )
        self.odomController = Controller("odom_controller", 'f1tenth', 0.1)

    def pubState(self, str):
        msg = String()
        msg.data = str
        self.state_pub.publish(msg)

    def getCurrState(self):
        return self.currState
    
    def changeState(self, newState):
        self.currState = newState

    def getOdom(self):
        state = self.odomController.get_observation("stateMachine")
        return state[0:2]

def main():
    rclpy.init()
    state_machine = StateMachine()
    print("In state machine")

    # Get initial odometry
    time.sleep(0.2)
    while(len(state_machine.odom) == 0):
        state_machine.odom = state_machine.getOdom()
        time.sleep(2)
    state_machine.init_odom = state_machine.odom
    state_machine.get_logger().info(str(state_machine.init_odom))
    #Waiting to move
    state_machine.odom = state_machine.getOdom()
    while (absoluteDistance(np.array(state_machine.init_odom), np.array(state_machine.odom)) < 0.2):
        state_machine.odom = state_machine.getOdom()
        time.sleep(0.1)
    state_machine.changeState("mapping")
    state_machine.get_logger().info("Begin mapping")
    time.sleep(5)
    state_machine.odom = state_machine.getOdom()
    while (absoluteDistance(np.array(state_machine.init_odom), np.array(state_machine.odom)) > 0.5):
        stringToPrint = "Initial odom" + str(state_machine.init_odom) + " , current odom: " + str(state_machine.odom) + ", distance: " + str(absoluteDistance(np.array(state_machine.init_odom), np.array(state_machine.odom)))
        state_machine.get_logger().info(stringToPrint) 
        state_machine.odom = state_machine.getOdom()
        time.sleep(0.1)
    stringToPrint = "Initial odom" + str(state_machine.init_odom) + " , current odom: " + str(state_machine.odom) + ", distance: " + str(absoluteDistance(np.array(state_machine.init_odom), np.array(state_machine.odom)))
    state_machine.get_logger().info(stringToPrint)   
    state_machine.get_logger().info("One lap completed")
    state_machine.changeState("planning")

    state_machine.destroy_node()
    rclpy.shutdown()
        


if __name__ == '__main__':
    main()