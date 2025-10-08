from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class F1TenthReset(Node):

    def __init__(self, env_name):
        name = env_name + '_reset'
        super().__init__(name)

        srv_cb_group = MutuallyExclusiveCallbackGroup()
        self.srv = self.create_service(Reset, name, callback=self.service_callback, callback_group=srv_cb_group)

        set_pose_cb_group = MutuallyExclusiveCallbackGroup()
        self.set_pose_client = self.create_client(
            SetEntityPose,
            f'world/empty/set_pose',
            callback_group=set_pose_cb_group
        )

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')


