from .ImageGraph import ImageGraph
from .PriorityQueue import PriorityQueue
from .util import heuristic

class AStar:
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
            
            for next in graph.neighbors(current):
                new_cost = cost_so_far[current] + graph.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(next, self.goal)
                    frontier.put(next, priority)
                    came_from[next] = current
        
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