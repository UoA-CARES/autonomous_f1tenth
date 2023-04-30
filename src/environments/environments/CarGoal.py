from simulation.simulation_services import SimulationServices
import rclpy
from ament_index_python import get_package_share_directory
import os
import xacro

rclpy.init()

def main():
    services = SimulationServices('empty')

    pkg_environments = get_package_share_directory('environments')
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    xacro_file = os.path.join(pkg_f1tenth_description, 'urdf', 'robot.urdf.xacro')
    robot_description = xacro.process_file(xacro_file)
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/goal.sdf", pose=[1, 1, 1])
    services.spawn(sdf=robot_description.toxml(), name='f1tenth')
    


if __name__ == '__main__':
    main()