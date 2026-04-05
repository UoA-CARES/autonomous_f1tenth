import sys
import rclpy
from rclpy.node import Node
from rclpy.task import Future

from ros_gz_interfaces.srv import SpawnEntity, DeleteEntity, SetEntityPose, ControlWorld
from ros_gz_interfaces.msg import Entity, EntityFactory, WorldControl, WorldReset
from geometry_msgs.msg import Pose, Point, Quaternion


class SimulationServices(Node):
    def __init__(self, world_name):
        super().__init__('simulation_services')

        self.world_name = world_name

        # Spawn Service --------------------------------------
        self.spawn_client = self.create_client(
            SpawnEntity,
            f'world/{self.world_name}/create',
        )

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('spawn service not available, waiting again...')

        self.spawn_future = Future()

        # Delete Service --------------------------------------
        self.delete_client = self.create_client(
            DeleteEntity,
            f'world/{self.world_name}/remove',
        )
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('delete service not available, waiting again...')

        self.delete_future = Future()

        # Set Pose Service --------------------------------------
        self.set_pose_client = self.create_client(
            SetEntityPose,
            f'world/{self.world_name}/set_pose',
        )
        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')

        self.set_pose_future = Future()

        # Reset Service --------------------------------------
        self.reset_client = self.create_client(
            ControlWorld,
            f'world/{self.world_name}/control',
        )

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset service not available, waiting again...')

        self.reset_future = Future()

    def spawn(self, *, sdf=None, sdf_filename=None, name='cool_car', pose=None, orientation=None):
        """
        Spawns Entity inside simulation

        :param sdf: sdf or urdf in string form
        :param sdf_filename: the path to your sdf or urdf file
        :param name: desired name of the entity
        :param pose: [x_pos, y_pos, z_pos]: desired x, y, and z position for entity to spawn

        :return: nothing
        """
        if sdf and sdf_filename:
            raise Exception('You passed both an sdf, and a path; pass only ONE')

        if pose is None:
            pose = [0, 0, 0]

        if orientation is None:
            orientation = [0, 0, 0, 0]

        if len(pose) != 3:
            raise Exception(f'Expected Pose length is 3, you gave {len(pose)}')

        request = SpawnEntity.Request()

        request.entity_factory = EntityFactory()

        if sdf:
            request.entity_factory.sdf = sdf
        elif sdf_filename:
            request.entity_factory.sdf_filename = sdf_filename

        x, y, z = pose
        request.entity_factory.pose.position.x = float(x)
        request.entity_factory.pose.position.y = float(y)
        request.entity_factory.pose.position.z = float(z)

        q_x, q_y, q_z, q_w = orientation
        request.entity_factory.pose.orientation.x = float(q_x)
        request.entity_factory.pose.orientation.y = float(q_y)
        request.entity_factory.pose.orientation.z = float(q_z)
        request.entity_factory.pose.orientation.w = float(q_w)

        request.entity_factory.name = name

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('spawn service not available, waiting again...')

        self.spawn_future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self, self.spawn_future)

        return self.spawn_future.result()

    def delete_entity(self, name, entity_type):
        """
        Deletes an entity from gazebo simulation

        :param name: name of the entity
        :param entity_type: int: type of entity using the following scheme
            NONE      = 0
            LIGHT     = 1
            MODEL     = 2
            LINK      = 3
            VISUAL    = 4
            COLLISION = 5
            SENSOR    = 6
            JOINT     = 7
        :return: none
        """

        request = DeleteEntity.Request()
        request.entity = Entity()

        request.entity.name = name
        request.entity.type = entity_type

        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('delete service not available, waiting again...')

        self.delete_future = self.delete_client.call_async(request)
        rclpy.spin_until_future_complete(self, self.delete_future)

        return self.delete_future.result()

    def reset(self):
        """
        Calls Reset Gazebo Service
        """

        request = ControlWorld.Request()
        request.world_control = WorldControl()
        request.world_control.reset = WorldReset()
        request.world_control.reset.all = True

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('delete service not available, waiting again...')

        self.reset_future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, self.reset_future)

        return self.reset_future.result()

    def set_pose(self, name, position=None, orientation=None, id=None, type=2):
        """
        Calls the SetEntityPose gazebo service
        :param entity_type: int: type of entity using the following scheme
            NONE      = 0
            LIGHT     = 1
            MODEL     = 2
            LINK      = 3
            VISUAL    = 4
            COLLISION = 5
            SENSOR    = 6
            JOINT     = 7
        """
        if position is None:
            position = [0, 0, 0]

        if orientation is None:
            orientation = [0, 0, 0, 0]

        x, y, z = position
        q_x, q_y, q_z, q_w = orientation

        request = SetEntityPose.Request()

        request.entity = Entity()
        request.entity.name = name
        request.entity.type = type

        request.pose = Pose()
        request.pose.position = Point()

        request.pose.position.x = float(x)
        request.pose.position.y = float(y)
        request.pose.position.z = float(z)

        request.pose.orientation = Quaternion()
        request.pose.orientation.x = float(q_x)
        request.pose.orientation.y = float(q_y)
        request.pose.orientation.z = float(q_z)
        request.pose.orientation.w = float(q_w)

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')

        self.set_pose_future = self.set_pose_client.call_async(request)
        rclpy.spin_until_future_complete(self, self.set_pose_future)

        return self.set_pose_future.result()