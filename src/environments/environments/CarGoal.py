from simulation.simulation_services import SimulationServices
import rclpy
from ament_index_python import get_package_share_directory

rclpy.init()

def main():
    services = SimulationServices('empty')

    sdf = get_package_share_directory('environments')

    services.spawn(f"{}")
    


if __name__ == '__main__':
    main()