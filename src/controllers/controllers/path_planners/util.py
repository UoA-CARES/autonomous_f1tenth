import numpy as np


def heuristic(a, b):
    #(x1, y1) = a
    #(x2, y2) = b
    #return abs(x1 - x2) + abs(y1 - y2) #Manhattan distance
    return np.linalg.norm(np.array(a) - np.array(b)) #changed to Euclidean distance