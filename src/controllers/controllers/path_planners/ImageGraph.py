import numpy as np

class ImageGraph:
    def __init__(self, image):
        self.image = image
    
    def neighbors(self, id):
        x, y = id
        neighbors = []
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (1, 0), (1, -1), (0, -1), (0, 1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.image.shape[0] and 0 <= ny < self.image.shape[1] and self.image[nx, ny] == 255:
                neighbors.append((nx, ny))
        return neighbors
    
    def cost(self, from_id, to_id):
        return np.sqrt((from_id[0] - to_id[0]) ** 2 + (from_id[1] - to_id[1]) ** 2)