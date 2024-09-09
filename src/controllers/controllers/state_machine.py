import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from message_filters import Subscriber
from nav_msgs.msg import Odometry

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
    
        self.currState = 'mapping'
        self.odom = []

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.odom = [position.x, position.y, position.z]
        #self.get_logger().info("reading odom")
    
    def pubState(self, str):
        msg = String()
        msg.data = str
        self.state_pub.publish(msg)

    def getCurrState(self):
        return self.currState
    
    def moveState(self, newState):
        self.currState = newState

def main():
    rclpy.init()
    state_machine = StateMachine()
    print("In state machine")
    
    while(1):
        rclpy.spin_once(state_machine)
        state_machine.get_logger().info(str(state_machine.odom))
        time.sleep(2)

    state_machine.destroy_node()
    rclpy.shutdown()
        


if __name__ == '__main__':
    main()