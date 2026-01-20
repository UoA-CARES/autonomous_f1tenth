import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from .util import absoluteDistance
import numpy as np
from .controller import Controller
from .mapping import MinimalClientAsync
import os

class StateMachine(Node):
    def __init__(self, startStage):
        super().__init__('state_machine')
        
        self.init_odom = []
        self.odom = []

        self.state_pub = self.create_publisher(
            String,
            '/state',
            10
        )
        self.odomController = Controller("odom_controller", 'f1tenth', 0.1)
        self.currState = startStage
        self.pubState(self.currState[0])

    def pubState(self, str):
        msg = String()
        msg.data = str
        self.state_pub.publish(msg)

    def getCurrState(self):
        return self.currState
    
    def changeState(self, newState):
        self.currState = newState
        self.pubState(self.currState[0])

    def getOdom(self):
        state = self.odomController.get_observation("stateMachine")
        return state[0:2]
    
    def initState(self):
        print("Init state")
        self.mapSaver = MinimalClientAsync()
        print("In state machine")

        # Get initial odometry
        time.sleep(0.2)
        while(len(self.odom) == 0):
            self.odom = self.getOdom()
            time.sleep(2) #reconsider length and position
        self.init_odom = self.odom
        while ((self.odom[0] > 2.9)& (self.odom[0] < 3.1)):
            self.odom = self.getOdom()
            time.sleep(0.1)
        self.init_odom = self.odom
        self.get_logger().info(str(self.init_odom))

        #Waiting to move
        self.odom = self.getOdom()
        while (absoluteDistance(np.array(self.init_odom), np.array(self.odom)) < 0.2):
            self.odom = self.getOdom()
            time.sleep(0.1) #reconsider position
        self.changeState("mapping")
        

    def mappingState(self):
        print("Mapping state")
        self.get_logger().info("Begin mapping")
        time.sleep(5)
        self.odom = self.getOdom()
        while (absoluteDistance(np.array(self.init_odom), np.array(self.odom)) > 1.2):
            stringToPrint = "Initial odom" + str(self.init_odom) + " , current odom: " + str(self.odom) + ", distance: " + str(absoluteDistance(np.array(self.init_odom), np.array(self.odom)))
            self.get_logger().info(stringToPrint) 
            self.odom = self.getOdom()
            time.sleep(0.1)
        stringToPrint = "Initial odom" + str(self.init_odom) + " , current odom: " + str(self.odom) + ", distance: " + str(absoluteDistance(np.array(self.init_odom), np.array(self.odom)))
        self.get_logger().info(stringToPrint)   
        self.get_logger().info("One lap completed")
        response = self.mapSaver.send_request('stateMap')

        self.mapSaver.destroy_node()
        self.changeState("planning")

    def planningState(self):
        print("Planning state")
        while(os.path.isfile('newpath.txt') == False):
            time.sleep(0.2)
        self.changeState("tracking")

    def trackingState(self):
        print("Tracking state")   
        while(1):
            time.sleep(1)
        self.changeState("end")

def main():
    rclpy.init()
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('startStage', 'init')
        ]
    )
    params = param_node.get_parameters(['startStage'])
    params = [param.value for param in params]
    startStage = params[0]
    
    state_machine = StateMachine(startStage)
    while(1):
        time.sleep(1)
    while(1):
        match(state_machine.getCurrState()):
            case "init":
                state_machine.initState()
            case "mapping":
                state_machine.mappingState()
            case "planning":
                state_machine.planningState()
            case "tracking":
                state_machine.trackingState()
            case "end":
                break
            case _:
                raise Exception("State machine error")
    state_machine.destroy_node()
    rclpy.shutdown()
        


if __name__ == '__main__':
    main()