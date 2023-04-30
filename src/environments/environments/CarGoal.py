from simulation.simulation_services import SimulationServices
import rclpy
from ament_index_python import get_package_share_directory

rclpy.init()

def main():
    services = SimulationServices('empty')

    pkg_environments = get_package_share_directory('environments')
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/goal.sdf")
    services.spawn(sdf_filename=f"{pkg_f1tenth_description}/sdf/goal.sdf")
    


if __name__ == '__main__':
    main()