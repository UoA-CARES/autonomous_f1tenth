from .ImageGraph import ImageGraph
from .PriorityQueue import PriorityQueue
import numpy as np

class AStar: #Need to follow DStar Lite structure more
    def __init__(self, start, goal, image):
        self.start = start
        self.goal = goal
        self.image = image

    def get_path(self):
        graph = ImageGraph(self.image)
        frontier = PriorityQueue()
        frontier.put(self.start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[self.start] = None
        cost_so_far[self.start] = 0
        
        while not frontier.empty():
            current = frontier.get()
            
            if current == self.goal:
                break
            lowest_cost = np.inf
            for next in graph.neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next, self.goal) #Should rename, priority implies a high number should be considered first but this is not the case.
                    frontier.put(next, priority)
                    if (priority < lowest_cost):
                        came_from[next] = current # Need to reevaluate because alg should be able to switch nodes to go for the most efficient path.....
                        lowest_cost = priority
        
        # If goal not reached, return None
        if self.goal not in came_from:
            return None
        
        # Reconstruct path
        path = []
        current = self.goal
        while current != self.start:
            path.append(current)  # Append each node to the path list
            current = came_from[current]
        path.append(self.start)
        path.reverse()  # Reverse the path to start from the start node
        
        return path  # Return the list of nodes representing the path
    
    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))