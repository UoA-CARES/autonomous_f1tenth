from simulation.simulation_services import SimulationServices #, ResetServices
from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
import rclpy
from ament_index_python import get_package_share_directory
import time

def main():
    rclpy.init()
    
    # Share Directories

    time.sleep(3)
    # env = CarGoalEnvironment('f1tenth')
    env = CarWallEnvironment('f1tenth')
    

    env.set_velocity(1, 1)

    state, _ = env.reset()

    while True:
        next_state, reward, terminated, truncated, info = env.step([1, 0])

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
            state, _ = env.reset()


if __name__ == '__main__':
    main()