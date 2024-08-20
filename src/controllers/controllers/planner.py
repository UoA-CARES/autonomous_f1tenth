import numpy as np
import cv2
import csv
import heapq
from .path_planners.DStarLite import DStarLite
from .path_planners.PriorityQueue import PriorityQueue
from .path_planners.util import heuristic
from.path_planners.ImageGraph import ImageGraph


def main():
    print("In Planner")
    # Read the PGM file
    image = cv2.imread('austin_1_save.pgm', cv2.IMREAD_GRAYSCALE)

    # Open CSV file in write mode with 'newline=""'
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
    
        # Coordinates of the dot (start position on the starting line)
        dot_x, dot_y = 480, 480  # Change these values to the coordinates of your dot

        # Edge Detection
        edges = cv2.Canny(thresholded_image, 50, 150, apertureSize=3)
    
        # Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
        if lines is not None:
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
            line2 = closest_points[1]
        
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
    
        # Dilation (to adjust for thickness of car)
        kernel = np.ones((5, 5), np.uint8)  # Kernel for dilation
        dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)
        
        # Draw the perpendicular line
        cv2.line(dilated_image, (perp_x1, perp_y1), (perp_x2, perp_y2), 255, 2)
        
        # Coordinates of the pixel to change (starting position on start line)
        x, y = 480, 480  # Change these values to the coordinates you want to modify
        add_car_image = dilated_image
    
        # Calculate points behind and in front of the starting line with an offset
        offset = 3  # Adjust this offset as needed, note thickness of car

        behind_x = int(dot_x - (dir_x * offset))
        behind_y = int(dot_y - (dir_y * offset))
        front_x = int(dot_x + (dir_x * offset))
        front_y = int(dot_y + (dir_y * offset))

        # Define the start and goal positions
        start = (front_y, front_x)
        goal = (behind_y, behind_x)

        # Thresholding
        threshold_value = 127  # Example threshold value
        max_value = 255         # Maximum pixel value after thresholding
        ret, add_car_image = cv2.threshold(add_car_image, threshold_value, max_value, cv2.THRESH_BINARY_INV)
        
        dstar = DStarLite(start, goal, add_car_image)
       
        # Run A* algorithm
        path = dstar.get_path()     #a_star(add_car_image, start, goal)
        if path is None:
            print(f"No valid path")
            exit()
    
        print(f"Path size:", len(path))
    
        output_image = add_car_image.copy()
        
        # Write each node (pixel coordinate) to the CSV file in separate columns x and y
        for pos in path:
            csv_writer.writerow([pos[1], pos[0]])  # Write x and y in separate columns
            output_image[pos[0], pos[1]] = 128
    
        
        cv2.imshow('Final Image', output_image)
        cv2.waitKey(3000)

        cv2.destroyAllWindows()

    # Close the CSV file
    file.close()



def a_star(image, start, goal):
    graph = ImageGraph(image)
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current
    
    # If goal not reached, return None
    if goal not in came_from:
        return None
    
    # Reconstruct path
    path = []
    current = goal
    while current != start:
        path.append(current)  # Append each node to the path list
        current = came_from[current]
    path.append(start)
    path.reverse()  # Reverse the path to start from the start node
    
    return path  # Return the list of nodes representing the path




if __name__ == '__main__':
    main()