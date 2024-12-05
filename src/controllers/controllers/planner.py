import numpy as np
import cv2
import csv
import rclpy
import time
import os
from .util import absoluteDistance
import yaml


def main():

    rclpy.init()
    
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('alg', 'random'),
            ('map', 'random'),
            ('yaml_path', 'random')
        ]
    )
    
    params = param_node.get_parameters(['alg', 'map', 'yaml_path'])
    params = [param.value for param in params]
    ALG = params[0]
    MAP = params[1]
    YAML_PATH = params[2]
    
    while (os.path.isfile(MAP) == False):
        time.sleep(1)

    data = yaml.load(open(YAML_PATH), Loader=yaml.SafeLoader)
    origin = data['origin']
    resolution = data['resolution']
    
    # Read the PGM file
    image = cv2.imread(MAP, cv2.IMREAD_GRAYSCALE)
    for row in range(image.shape[0]):
        if image[row, 0] == 254:
            image[row, 0] = 0
        if image[row, image.shape[1]-1] == 254:
            image[row, image.shape[1]-1] = 0
        
    for col in range(image.shape[1]):
        if image[0, col] == 254:
            image[0, col] = 0
        if image[image.shape[0]-1, col] == 254:
            image[image.shape[0]-1, col] = 0
    file = open("output.csv", 'w', newline='')
    
    csv_writer = csv.writer(file)

    # Check if the image was loaded correctly
    if image is None:
        print("Error: Could not open or find the image.")
    else:
        # Thresholding
        threshold_value = 127  # Example threshold value
        max_value = 255         # Maximum pixel value after thresholding
        ret, thresholded_image = cv2.threshold(image, threshold_value, max_value, cv2.THRESH_BINARY_INV)
        dot_x, dot_y = 0, 0  # Change these values to the coordinates of your dot

        # Edge Detection
        edges = cv2.Canny(thresholded_image, 50, 150, apertureSize=3)
    
        # Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 115)
        #lines = []
        if 1 == 0:
            # Convert Hough lines to endpoints
            lines = [((rho, theta)) for rho, theta in lines[:, 0]]
        
            # Find the closest points on the lines to the dot
            closest_points = []
            for rho, theta in lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
            
                # Find the closest point on the line segment to the dot
                dist = abs((y2 - y1) * dot_x - (x2 - x1) * dot_y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                closest_points.append(((x1, y1), (x2, y2), dist))
        
            # Sort the lines by the distance to the dot
            closest_points = sorted(closest_points, key=lambda x: x[2])
        
            # Choose the two closest lines (ie. the track lines)
            line1 = closest_points[0]
            #line2 = closest_points[1]
        
            # Calculate the midpoint of the dot
            mid_x = dot_x
            mid_y = dot_y
        
            # Calculate the direction of the perpendicular line
            dir_x = line1[1][1] - line1[0][1]
            dir_y = line1[0][0] - line1[1][0]
        
            # Normalize the direction
            length = np.sqrt(dir_x ** 2 + dir_y ** 2)
            dir_x /= length
            dir_y /= length
        
            # Calculate the perpendicular direction (rotate by 90 degrees)
            perp_dir_x = -dir_y
            perp_dir_y = dir_x

            # Calculate the endpoints of the perpendicular line
            perp_line_length = 20  # Length of the perpendicular line
            perp_x1 = int(mid_x + perp_line_length * perp_dir_x)
            perp_y1 = int(mid_y + perp_line_length * perp_dir_y)
            perp_x2 = int(mid_x - perp_line_length * perp_dir_x)
            perp_y2 = int(mid_y - perp_line_length * perp_dir_y)
        else:
            print("No lines")
            origin = np.asarray([origin[0], origin[1]])
            shape = image.shape
            resolution = resolution
            start = findOrigin(origin, shape, resolution)
            #Only for track 1
            start[0] = start[0] +5
            if int(image[start[0], start[1]]) != 254:
                print("Not possible")
            goalx = start[0] -3
            goaly = start[1] - 3
            goal = [goalx, goaly]
            
        # Dilation (to adjust for thickness of car)
        kernel = np.ones((15, 15), np.uint8)  # Kernel for dilation 15, 15, for track2
        dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)
        colour = 0
        i = start[0]
        while colour == 0:
            dilated_image[i, start[1]-1] = 254
            i +=1
            colour = dilated_image[i, start[1]-1]
        colour = 0
        i = start[0]-1
        while colour == 0:
            dilated_image[i, start[1]-1] = 254
            i-=1
            colour = dilated_image[i, start[1]-1]
        # Draw the perpendicular line
        # dilated_image[start[0], start[1]] = 128
        # dilated_image[start[0]+1, start[1]+1] = 128
        # dilated_image[start[0]-1, start[1]-1] = 128
        # dilated_image[start[0], start[1]+1] = 128
        # dilated_image[start[0], start[1]-1] = 128
        # dilated_image[start[0]+1, start[1]] = 128
        # dilated_image[start[0]-1, start[1]] = 128
        # dilated_image[start[0]-1, start[1]+1] = 128
        #cv2.line(dilated_image, start, newPoint, 150, 4)
        # dilated_image[goal[0], goal[1]] = 200
        # dilated_image[goal[0]+1, goal[1]+1] = 200
        # dilated_image[goal[0]-1, goal[1]-1] = 200
        # dilated_image[goal[0], goal[1]+1] = 200
        # dilated_image[goal[0], goal[1]-1] = 200
        # dilated_image[goal[0]+1, goal[1]] = 200
        # dilated_image[goal[0]-1, goal[1]] = 200
        # dilated_image[goal[0]-1, goal[1]+1] = 200
        goal = (goalx, goaly)
        start = (start[0], start[1])
        # Coordinates of the pixel to change (starting position on start line)
        x, y = 480, 480  # Change these values to the coordinates you want to modify
        add_car_image = dilated_image
    
        # Calculate points behind and in front of the starting line with an offset
        offset = 3  # Adjust this offset as needed, note thickness of car

        # behind_x = int(dot_x - (dir_x * offset))
        # behind_y = int(dot_y - (dir_y * offset))
        # front_x = int(dot_x + (dir_x * offset))
        # front_y = int(dot_y + (dir_y * offset))

        # Define the start and goal positions
        # start = (front_y, front_x)
        # goal = (behind_y, behind_x)

        # Thresholding
        threshold_value = 127  # Example threshold value
        max_value = 255         # Maximum pixel value after thresholding
        ret, add_car_image = cv2.threshold(add_car_image, threshold_value, max_value, cv2.THRESH_BINARY_INV)
        policy = policy_factory(ALG, start, goal, add_car_image)
        path = policy.get_path()
        if path is None:
            print(f"No valid path")
            exit()
    
        output_image = add_car_image.copy()
        
        # Write each node (pixel coordinate) to the CSV file in separate columns x and y
        for pos in path:
            csv_writer.writerow([pos[1], pos[0]])  # Write x and y in separate columns
            output_image[pos[0], pos[1]] = 128

        cv2.imwrite('path.pgm', output_image)
        cv2.destroyAllWindows()

        origin = np.asarray([origin[0], origin[1]])
        shape = output_image.shape
        resolution = resolution
        newcoords = coordinateShift(path, origin, shape, resolution)
        newcoords = trimCoords(newcoords, 1)
        newPath = open("newpath.txt", 'w')
        for state in (newcoords):
            s = '['+str(round(state[0], 2))+', '+str(round(state[1], 2)) + '], '
            newPath.write(s)

    # Close the CSV file
    file.close()
    newPath.close()

def policy_factory(alg, start, goal, add_car_image):
    policy = 0
    match alg:
        case 'astar':
            from .path_planners.AStar import AStar
            policy = AStar(start, goal, add_car_image)
            return policy
        case 'dstarlite':
            from .path_planners.DStarLite import DStarLite
            policy = DStarLite(start, goal, add_car_image)
            return policy
        case _:
            return policy

def coordinateShift(path, origin, shape, resolution):
    mx = resolution
    my = -1*resolution
    coordinates = []
    ymax = origin[1]+shape[0]*resolution
    for pos in path:
        newposx = mx*pos[1] + origin[0]
        newposy = my*pos[0] + ymax
        coordinates.append([newposx, newposy])
    return coordinates

def trimCoords(path, minDist):
    coordinates = []
    prev = np.asarray([np.inf, np.inf])
    path = np.array(path)
    for pos in path:
        distance = absoluteDistance(pos, prev)
        if distance > minDist:
            coordinates.append(pos)
            prev = pos
        elif (np.all(pos == path[-1]) & (absoluteDistance(pos, path[0]) > minDist)):
            coordinates.append(pos)
    return coordinates

def findOrigin(origin, shape, resolution):
    mx = resolution
    my = -1*resolution
    ymax = origin[1]+shape[0]*resolution
    row = (3.5-ymax)/my 
    col = (5-origin[0])/mx
    return [int(row), int(col)]

if __name__ == '__main__':
    main()
