import numpy as np

# Returns steering angle to turn to goal
def turn_to_goal(location, yaw, goal, goal_tolerance=0.5, angle_diff_tolerance=0.1, max_turn=0.85):

    distance = goal - location # x, y array

    if ((abs(distance[0]) < goal_tolerance) and (abs(distance[1] < goal_tolerance))): # Already at goal
        ang = 0
        return ang
     
    angle_to_goal = np.arctan2(distance[1], distance[0])
    if (((angle_to_goal - yaw) > angle_diff_tolerance) or ((angle_to_goal - yaw) < -angle_diff_tolerance)):
            
        # take the shortest turning angle
        ang = angle_to_goal - yaw
        if ang > np.pi:
            ang -= 2 * np.pi
        elif ang < -np.pi:
            ang += 2 * np.pi


        # make sure turning angle is not more than 90deg
        if ang > max_turn:
            ang = max_turn
        elif ang < -1*max_turn:
            ang = -1*max_turn
        return ang
    else:
         return 0
    
def absoluteDistance(point1, point2):
    distance = point2 - point1
    hyp = np.hypot(distance[0], distance[1])
    return hyp


def closestPointInd(location, path):
    index = -1
    minDist = np.inf
    row, _ = path.shape
    for i in range(row):
        distance = absoluteDistance(location, path[i])
        if (distance < minDist):
            minDist = distance
            index = i
    return index

def closestPointIndAhead(location, path):
    closestPointIndex = closestPointInd(location, path)
    closestPoint = path[closestPointIndex]
    nextPoint = path[closestPointIndex+1]
    if (absoluteDistance(location, nextPoint) < absoluteDistance(closestPoint, nextPoint)):
        return closestPointIndex + 1
    else:
        return closestPointIndex