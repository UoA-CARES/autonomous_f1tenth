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
    coordinates = ([0, 0], [1, 0], [3, 1], [5, 2]) #placeholder values
    return coordinates

def budapestLap():
    coordinates = ([0, 0], [1, 0], [3, 1], [5, 2]) #placeholder values
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