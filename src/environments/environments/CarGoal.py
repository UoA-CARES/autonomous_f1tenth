from simulation.simulation_services import SimulationServices #, ResetServices
import rclpy
from ament_index_python import get_package_share_directory
import os
import xacro
import sys
import logging
import time
from .CarGoalEnvironment import CarGoalEnvironment

def main():
    rclpy.init()
    
    # Share Directories
    pkg_environments = get_package_share_directory('environments')
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')

    # F1tenth Car SDF
    xacro_file = os.path.join(pkg_f1tenth_description, 'urdf', 'robot.urdf.xacro')
    robot_description = xacro.process_file(xacro_file)

    services = SimulationServices('empty')

    services.spawn(sdf_filename=f"{pkg_environments}/sdf/goal.sdf", pose=[1, 1, 1], name='goal')

    time.sleep(3)
    cargoal = CarGoalEnvironment('f1tenth')

    cargoal.set_velocity(1, 1)
    
    while True:
        cargoal.reset()


    



    


if __name__ == '__main__':
    main()