def straightLine():
    coordinates = ([0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0]) #placeholder values
    return coordinates

def circleCW():
    radius = 5
    coordinates = ([0, 0], [radius, -1*radius], [0, -2*radius], [-1*radius, -1*radius]) #placeholder values
    coordinates = ([0, 0], [radius*3/5, radius*-1/5], [radius*4/5, radius*-2/5], [radius, -radius], [radius*4/5, radius*-8/5], [radius*3/5, radius*-9/5], [0, -2*radius], [radius*-3/5, radius*-9/5], [radius*-4/5, radius*-8/5], [-1*radius, -radius], [radius*-4/5, radius*-2/5], [radius*-3/5, radius*-1/5], [0, 0])
    return coordinates

def circleCCW():
    radius = 5
    coordinates = ([0, 0], [radius*3/5, radius*1/5], [radius*4/5, radius*2/5], [radius, radius], [radius*4/5, radius*8/5], [radius*3/5, radius*9/5], [0, 2*radius], [radius*-3/5, radius*9/5], [radius*-4/5, radius*8/5], [-1*radius, radius], [radius*-4/5, radius*2/5], [radius*-3/5, radius*1/5], [0, 0]) #placeholder values
    return coordinates


def austinLap():
    #Approximate centre points of Austin circuit
    coordinates = ([3, -2.5], [6, -4.5], [10, -7.5], [14, -10.5], [18, -14], [119, 48], [115, 47], [111, 46], [107, 44.5], [103, 43.5], [99, 42.5], [95, 41.5], [91, 40.5], [87, 39.5], [83, 39], [79, 38], [75, 37.5], [71, 37], [67, 36.5], [63, 35.5], [59, 35], [55, 34.5], [51, 34], [47, 34], [43, 33.5], [41.5, 32], [43.5, 29], [45.5, 26], [48.5, 22], [50, 18], [47, 17.5], [43, 18], [41, 22], [38.5, 25], [35, 26], [33, 26], [34, 23], [36, 19], [38.5, 15], [40.5, 11], [40, 9], [38.5, 5], [35.5, 3], [32, 1], [28, 1], [24, 3], [21, 6], [18.5, 9], [16, 12.5], [14, 15], [12, 16.5], [8, 16], [0, 12.5], [-4, 10.5], [-8, 8.5], [-9, 7], [-5, 4], [-1, 0.5]) #placeholder values
    return coordinates

def budapestLap():
    coordinates = ([0, 0], [4, -3.5], [8, -6.5], [12, -10], [16, -12], [20, -11], [22, -9], [22, -5], [18, 0], [14, 3], [10, 7], [12, 11.5], [16, 11.5], [20, 8], [24.63, 3.08], [28, -1], [31, -1], [34, 3], [37, 6], [40, 9], [44, 13.5], [48, 17.5], [52, 22], [54, 25], [54.5, 29], [52.5, 33], [50, 37], [48, 41], [47.5, 43], [47.5, 45], [48.5, 49], [49, 52], [49, 54], [49.5, 58], [49, 60], [47, 61.5], [44, 62], [40, 62.5], [38, 63], [36, 66], [35, 70], [34, 74], [33, 78], [34, 80], [35, 82], [33, 85], [29.5, 89], [26, 93], [21, 97], [17, 99], [13, 98], [10.5, 94], [11.5, 90], [12.5, 86], [13.5, 82], [14, 78], [14, 76], [10.5, 72], [8, 68], [5.5, 64], [3.5, 60], [1.5, 56], [-0.5, 52], [-2.5, 48], [-4.5, 44], [-6.5, 40], [-7, 38], [-5.5, 34], [-3.5, 30], [-2, 26], [-3, 23], [-6, 21.5], [-10, 23], [-14.5, 27], [-19, 31], [-24, 35], [-28, 37], [-32, 38.5], [-36, 39.5], [-40, 39.5], [-44, 38.5], [44, 37], [-42.5, 35], [-38, 31], [-34, 28], [-30, 24.5], [-26, 21.5], [-20, 16.5], [-16, 13], [-12, 10], [-8, 6.5], [-4, 3]) #placeholder values
    return coordinates

def hockenheimLap():
    coordinates = ([0, 0], [1, 0], [3, 1], [5, 2]) #placeholder values
    return coordinates

def melbourneLap():
    # TBC
    coordinates = ([0, 0], [1, 0], [3, 1], [5, 2]) #placeholder values
    return coordinates

def saopaoloLap():
    # TBC
    coordinates = ([0, 0], [1, 0], [3, 1], [5, 2]) #placeholder values
    return coordinates

def shanghaiLap():
    # TBC
    coordinates = ([0, 0], [1, 0], [3, 1], [5, 2]) #placeholder values
    return coordinates