import numpy as np

# Returns steering angle to turn to goal
def turn_to_goal(location, yaw, goal, goal_tolerance=0.5, angle_diff_tolerance=0.2, max_turn=1):

    distance = goal - location # x, y array
    if ((abs(distance[0]) < goal_tolerance) and (abs(distance[1] < goal_tolerance))): # Already at goal
        ang = 0
        return ang
     
    angle_to_goal = np.arctan2(distance[1], distance[0])
    if (((angle_to_goal - yaw) > angle_diff_tolerance) or ((angle_to_goal - yaw) < -angle_diff_tolerance)):
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

def closestPointIndAhead(location, path, buffer=0.8): #buffer = 1 for turn and drive
    closestPointIndex = closestPointInd(location, path)
    row, _ = path.shape
    if (absoluteDistance(location, path[closestPointIndex]) < buffer):
        if (closestPointIndex < (row-1)):
            closestPointIndex += 1
        else:
            closestPointIndex = 0
    closestPoint = path[closestPointIndex]
    try:    
        nextPoint = path[closestPointIndex+1]
    except:
        nextPoint = path[0]
    if (absoluteDistance(location, nextPoint) < absoluteDistance(closestPoint, nextPoint)):
        if (closestPointIndex < (row-1)):
            return closestPointIndex + 1
        else:
            return 0
    else:
        return closestPointIndex
    
def furthestPointInRange(location, path, look_ahead):
    row, _ = path.shape
    minDist = np.inf
    lastPointInd = -1
    for i in range(row):
        point = path[i]
        distance = point - location
        hyp = np.hypot(distance[0], distance[1])    
        if (hyp < minDist): # Find closest point on path to car
            closestPointInd = i
            minDist = hyp
        if (hyp < look_ahead): # Find last point within lookahead distance to car
            lastPointInd = i
    if lastPointInd == (range(row-1)):
        for i in range(2):
            point = path[i]
            distance = point - location
            hyp = np.hypot(distance[0], distance[1])
            if (hyp < look_ahead):
                lastPointInd = i
    if lastPointInd < 0: # If no points within lookahead range, goal1 is closest point
        return closestPointInd
    return lastPointInd

# Needs fix
def linCalc(ang, maxLin=1, maxAng=0.85, fullSpeedCutoff = 0.05):
    if ang < fullSpeedCutoff:
        return maxLin
    else:
        minLin = 0.05*maxLin
        # Calculate linear decreasing function
        gradient = (minLin - maxLin)/(maxAng - fullSpeedCutoff)
        c = minLin - (maxAng * gradient)
        lin = gradient*ang + c
        return lin
    

def loadPath(filename):
    path = []
    file = open(filename, "r")
    txt = file.read()
    file.close()
    i = 0
    while i < len(txt)-1:
        if txt[i] == '[': # New coordinate
            i+= 1
            string = ''
            while txt[i] != ',': # First value
                string += txt[i]
                i += 1
            num1 = float(string)
            string = ''
            i += 2
            while txt[i] != ']': # Second value
                string += txt[i]
                i+=1
            num2 = float(string)
        path.append([num1, num2])
        i += 1 
    path = np.array(path)
    return path