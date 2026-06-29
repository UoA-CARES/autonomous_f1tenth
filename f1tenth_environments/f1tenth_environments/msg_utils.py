from geometry_msgs.msg import Point, Pose
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose

from . import geometry_utils


def build_set_model_pose_request(
    *,
    model_name: str,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> SetEntityPose.Request:
    """Create a SetEntityPose request for a Gazebo model."""
    request = SetEntityPose.Request()
    request.entity = Entity()
    request.entity.name = model_name
    request.entity.type = Entity.MODEL

    request.pose = Pose()
    request.pose.position = Point()
    request.pose.position.x = float(x)
    request.pose.position.y = float(y)
    request.pose.position.z = float(z)

    orientation = geometry_utils.get_quaternion_from_euler(roll, pitch, yaw)
    request.pose.orientation.x = orientation[0]
    request.pose.orientation.y = orientation[1]
    request.pose.orientation.z = orientation[2]
    request.pose.orientation.w = orientation[3]
    return request
