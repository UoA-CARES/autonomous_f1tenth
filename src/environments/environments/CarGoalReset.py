from rclpy.node import Node
from std_srvs.srv import Trigger
import sys

print(sys.argv)

class ResetService(Node):
    """
    Calls the ResetServices class

    This makes the car move back to the original position which is [0,0,0]
       To make the car move back to original position, need to get current position of car into an array
       and get the car to move back from that position to [0,0,0]
    And this make the goal move to a random position on the grid
       To make the goal move to a random position, need to set max. x, y, z values of the grid and get the goal
       in a randomly generated postition

    Need clarification of where the current position is stored and if there is already an array for it 
    And where the maximum values of the x, y, z values of the grid are located 
    """

    """
    Reti's Notes:
    This is a reset service that does two things:
        - Moves the car back to 0, 0 
            To do this, use the SimulationServices class - it has a set pose method, that can set the pose of gazebo objects
        - Moves the goal to a random position
            Use the same service
    
    Notes:
        For now, we can hard code the boundaries of where you're spawning the goal - this is not an issue,
        as it can simply by an environmental factor
        
        To help a bit more, I've scaffolded what the service node should look like
    """
    
    def __init__(self):
        super()._init_('reset service')
        self.srv = self.create_service(srv_type=Trigger, srv_name='car_goal_reset', callback=self.reset)

        self.original_pos_car = [0,0,0]
       # x,y,z = self.current_pos_car #find current position of car 
        
    def reset(self):
        # Here, Implement the reset
        pass


def main():
    print('Reset Node')



if __name__ == '__main__':
    main()

    

#create services node
#make car move back to (0,0) position
#make goal move to a random postion