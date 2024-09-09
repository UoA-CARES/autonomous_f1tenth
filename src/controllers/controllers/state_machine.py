import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        self.state_pub = self.create_publisher(
            String,
            '/state',
            10
        )

    def pubState(self, str):
        msg = String()
        msg.data = str
        self.state_pub.publish(msg)

def main():
    rclpy.init()
    state_machine = StateMachine()
    print("In state machine")
    while(1):
        state_machine.pubState('w')
        time.sleep(2)

    state_machine.destroy_node()
    rclpy.shutdown()
        


if __name__ == '__main__':
    main()