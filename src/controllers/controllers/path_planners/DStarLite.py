import heapq
import numpy as np
from.PriorityQueue import PriorityQueue

class DStarLite:
    def __init__(self, start, goal, image):
        self.start = start
        self.goal = goal
        self.image = image
        self.g = {}
        self.rhs = {}
        self.U = PriorityQueue()
        self.km = 0
        self.initialize()

    def initialize(self):
        self.g[self.goal] = float('inf')
        self.rhs[self.goal] = 0
        self.U.put(self.goal, self.calculate_key(self.goal))
    
    def calculate_key(self, s):
        g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (g_rhs + self.heuristic(self.start, s) + self.km, g_rhs)
    
    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(self.g.get(neigh, float('inf')) + self.cost(u, neigh) for neigh in self.neighbors(u))
        if u in [x[1] for x in self.U.elements]:
            self.U.elements = [x for x in self.U.elements if x[1] != u]
            heapq.heapify(self.U.elements)
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            self.U.put(u, self.calculate_key(u))
    
    def neighbors(self, id):
        x, y = id
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.image.shape[0] and 0 <= ny < self.image.shape[1] and self.image[nx, ny] == 255:
                neighbors.append((nx, ny))
        return neighbors

    def cost(self, from_id, to_id):
        return np.sqrt((from_id[0] - to_id[0]) ** 2 + (from_id[1] - to_id[1]) ** 2)

    def compute_shortest_path(self):
        while (self.U.empty() == False and 
               (self.U.elements[0][0] < self.calculate_key(self.start) or self.rhs.get(self.start, float('inf')) != self.g.get(self.start, float('inf')))):
            u = self.U.get()
            if self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for neigh in self.neighbors(u):
                    self.update_vertex(neigh)
            else:
                self.g[u] = float('inf')
                for neigh in self.neighbors(u) + [u]:
                    self.update_vertex(neigh)

    def get_path(self):
        self.compute_shortest_path()
        path = []
        current = self.start
        while current != self.goal:
            path.append(current)
            min_cost = float('inf')
            next_node = None
            for neigh in self.neighbors(current):
                cost = self.g.get(neigh, float('inf')) + self.cost(current, neigh)
                if cost < min_cost:
                    min_cost = cost
                    next_node = neigh
            if next_node is None:
                break
            current = next_node
        path.append(self.goal)
        return path

    def update_graph(self, changed_edges):
        self.km += self.heuristic(self.start, self.goal)
        for (from_id, to_id, new_cost) in changed_edges:
            if new_cost == float('inf'):
                self.image[to_id[0], to_id[1]] = 0  # Mark as an obstacle
            else:
                self.image[to_id[0], to_id[1]] = 255  # Mark as traversable
            
            self.update_vertex(from_id)
            self.update_vertex(to_id)
        
        self.compute_shortest_path()

    def heuristic(self, a, b):
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)
    