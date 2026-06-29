import numpy as np
import random
from rclpy.impl import rcutils_logger
   
class Random():
    logger = rcutils_logger.RcutilsLogger(name="random_log")

    def __init__(self): 
        self.logger.info("-------------------------------------------------")
        self.logger.info("Random Alg created")
        self.multiCoord = False

    def select_action(self, state, goal):
        action = np.asarray([random.uniform(0, 3), random.uniform(-3.14, 3.14)])
        self.logger.info("DRIVE LIN_V: "+str(action[0]))
        self.logger.info("DRIVE ANGLE: "+str(action[1]))
        self.logger.info("-------------------------")
        return action
