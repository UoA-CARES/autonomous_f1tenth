import rclpy
from rclpy.node import Node


class SimulationSupervisor(Node):
    def __init__(self):
        super().__init__("sim")
        self.declare_parameters(
            "",
            [
                ("environment", "CarTrack"),
                ("algorithm", "ftg"),
                ("start_stage", "track"),
                ("track", "track_01"),
                ("path_file_path", "newpath.txt"),
            ],
        )
        self.get_logger().info("Simulation supervisor ready")


def main():
    rclpy.init()
    node = SimulationSupervisor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
