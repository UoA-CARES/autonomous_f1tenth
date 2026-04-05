# create simple ros2 subscriber and publisher
import random
import os
from multiprocessing import Process

import rclpy
from rclpy.node import Node
from launch import LaunchService, LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python import get_package_share_directory

from f1tenth_msgs.msg import Available, Configure

from statemachine import StateMachine, State


class StateController(Node, StateMachine):
    """
    StateController class

    States:
        - unconfigured
        - active

    Transitions:
        - configure
        - deactivate

    During unconfgured state:
        If it is sent a Available.msg what has get_available_cars = True on the "/available_cars"
        then it will publish its ID on the topic in reply

        When sent Configure.msg on "/configure_car" topic it will change to active state and config as
        the message says and transition to active state
    """

    # States
    unconfigured = State('unconfigured', initial=True)
    active = State('active')

    # Transitions
    configure = unconfigured.to(active)
    deactivate = active.to(unconfigured)

    def __init__(self):
        super().__init__('state_controller')

        # Create random ID
        self.id = random.randint(0, 1000)
        self.name: str | None = None

    # Transitions
    @unconfigured.enter
    def on_enter_unconfigured(self):
        # Create topics and callbacks
        self.get_logger().info("StateController: Entering unconfigured state")

        # Subscribers
        self.sub_available_cars = self.create_subscription(
            Available,
            '/available_cars',
            self.available_cars_callback,
            10
        )
        self.sub_configure_car = self.create_subscription(
            Configure,
            '/configure_car',
            self.configure_car_callback,
            10
        )

        # Publishers
        self.pub_available_cars = self.create_publisher(
            Configure,
            '/available_cars',
            10
        )

    @unconfigured.exit
    def on_exit_unconfigured(self):
        # Destroys subscribers and publishers
        self.destroy_subscription(self.sub_available_cars)
        self.destroy_subscription(self.sub_configure_car)
        self.destroy_publisher(self.pub_available_cars)

    @active.enter
    def on_enter_active(self):
        """Callback method to handle the transition into the 'active' state.

        This method is called when entering the 'active' state. It logs the state
        transition, initializes the hardware-related nodes using a LaunchService object,
        and starts a separate process to run the LaunchService.
        """
        self.log_state_transition()
        ls = self.create_launch_service()
        self.launch_hardware_bringup(ls)
        self.process = self.run_launch_service_in_new_process(ls)
        self.process.start()

    @active.exit
    def on_exit_active(self):
        """Callback method to handle the transition out of the 'active' state.

        This method is called when exiting the 'active' state. It terminates the
        hardware-related nodes launched in a separate process.
        """
        self.process.terminate()

    def log_state_transition(self):
        """Logs the state transition."""
        self.get_logger().info("StateController: Entering active state")

    def create_launch_service(self):
        """Creates and returns a new LaunchService object."""
        return LaunchService()

    def launch_hardware_bringup(self, launch_service):
        """Launches the 'hardware_bringup.launch.py' file using the provided LaunchService."""
        pkg_f1tenth_bringup = get_package_share_directory('f1tenth_bringup')

        launch_service.include_launch_description(
            LaunchDescription([
                IncludeLaunchDescription(
                    launch_description_source=PythonLaunchDescriptionSource(
                        os.path.join(pkg_f1tenth_bringup, 'launch', 'hardware_bringup.launch.py')),
                    launch_arguments={
                        "name": self.name,
                    }
                )
            ])
        )

    def run_launch_service_in_new_process(self, launch_service):
        """Runs the provided LaunchService in a new process and returns the process."""
        process = Process(target=launch_service.run)
        process.daemon = True
        return process

    # Callbacks
    def available_cars_callback(self, msg):
        """
        When sent Available.msg on "/available_cars" topic it will reply with its ID
        """
        if msg.get_available_cars:
            self.get_logger().info("StateController: Publishing car ID")
            self.pub_car_id.publish(Configure(id=self.id))

    def configure_car_callback(self, msg):
        """
        When sent Configure.msg on "/configure_car" topic it will change to active state and config as
        the message says and transition to active state
        """
        if msg.id != self.id:
            return

        self.get_logger().info("StateController: Received configure car message")
        self.get_logger().info("StateController: Transitioning to active state")

        self.name = msg.name
        self.configure()

        self.get_logger().info("StateController: Transitioned to active state")


def main(args=None):
    rclpy.init(args=args)
    state_controller = StateController()
    rclpy.spin(state_controller)
    state_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
