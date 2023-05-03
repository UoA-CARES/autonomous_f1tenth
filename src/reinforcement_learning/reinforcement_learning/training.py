from simulation.simulation_services import SimulationServices #, ResetServices
from environments.CarGoalEnvironment import CarGoalEnvironment
import rclpy
from ament_index_python import get_package_share_directory
import time

def main():
    rclpy.init()
    
    # Share Directories
    pkg_environments = get_package_share_directory('environments')
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')

    services = SimulationServices('empty')

    services.spawn(sdf_filename=f"{pkg_environments}/sdf/goal.sdf", pose=[1, 1, 1], name='goal')

    time.sleep(3)
    cargoal = CarGoalEnvironment('f1tenth')

    cargoal.set_velocity(1, 1)

    state, _ = cargoal.reset()

    while True:
        next_state, reward, terminated, truncated, info = cargoal.step([1, 1])

        print(
            f'State: {state}'
            f'\nNext State: {next_state}'
            f'\nReward: {reward}'
            f'\nTerminated: {terminated}'
            f'\nTruncated: {truncated}'
            f'\nInfo: {info}'
        )

        state = next_state

        if terminated or truncated:
            state, _ = cargoal.reset()


if __name__ == '__main__':
    main()