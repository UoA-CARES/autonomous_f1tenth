from environments import CarGoal

import rclpy

rclpy.init()

def main():
    print('Hi from reinforcement_learning.')

    env = CarGoal()


if __name__ == '__main__':
    main()
