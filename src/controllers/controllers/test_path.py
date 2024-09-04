import numpy as np

def straightLine():
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0]]) #placeholder values
    return coordinates

def circleCW():
    radius = 5
    coordinates = np.asarray([[0, 0], [radius*3/5, radius*-1/5], [radius*4/5, radius*-2/5], [radius, -radius], [radius*4/5, radius*-8/5], [radius*3/5, radius*-9/5], [0, -2*radius], [radius*-3/5, radius*-9/5], [radius*-4/5, radius*-8/5], [-1*radius, -radius], [radius*-4/5, radius*-2/5], [radius*-3/5, radius*-1/5], [0, 0]])
    return coordinates

def circleCCW():
    radius = 5
    coordinates = np.asarray([[0, 0], [radius*3/5, radius*1/5], [radius*4/5, radius*2/5], [radius, radius], [radius*4/5, radius*8/5], [radius*3/5, radius*9/5], [0, 2*radius], [radius*-3/5, radius*9/5], [radius*-4/5, radius*8/5], [-1*radius, radius], [radius*-4/5, radius*2/5], [radius*-3/5, radius*1/5], [0, 0]]) #placeholder values
    return coordinates


def austinLap():
    #Approximate centre points of Austin circuit
    coordinates = np.asarray([[3, -2.5], [6, -4.5], [10, -7.5], [14, -10.5], [18, -14], [21.88, -16.71], [23, -17.5], [25, -19], [27, -20.5], [29, -22], [31, -24], [33, -25.5], [35, -27], [38, -28.5], [40, -29], [41, -27.5], [39, -22], [37, -17], [36.5, -13.5], [37, -11.5], [38.5, -9], [40.5, -7], [43.5, -5], [50, -1], [53, 1.5], [55, 5], [57, 8], [60, 9.5], [62, 11], [63, 14], [63.5, 17], [65, 20], [67, 21.5], [70, 23], [73, 23], [76, 22], [79, 20.5], [82, 19.5], [85, 21.5], [88, 24.5], [91, 27], [94, 26.5], [96, 25], [99, 23.5], [104, 24.5], [108, 25.5], [113, 26.5], [115, 29], [118, 33], [121, 37], [123.5, 40], [126, 43], [128, 46], [130, 48], [131, 50], [129, 51], [126, 50.5], [122, 49], [119, 48], [115, 47], [111, 46], [107, 44.5], [103, 43.5], [99, 42.5], [95, 41.5], [91, 40.5], [87, 39.5], [83, 39], [79, 38], [75, 37.5], [71, 37], [67, 36.5], [63, 35.5], [59, 35], [55, 34.5], [51, 34], [47, 34], [43, 33.5], [41.5, 32], [43.5, 29], [45.5, 26], [48.5, 22], [50, 18], [47, 17.5], [43, 18], [41, 22], [38.5, 25], [35, 26], [33, 26], [34, 23], [36, 19], [38.5, 15], [40.5, 11], [40, 9], [38.5, 5], [35.5, 3], [32, 1], [28, 1], [24, 3], [21, 6], [18.5, 9], [16, 12.5], [14, 15], [12, 16.5], [8, 16], [0, 12.5], [-4, 10.5], [-8, 8.5], [-9, 7], [-5, 4], [-1, 0.5]]) #placeholder values
    return coordinates

def budapestLap():
    coordinates = np.asarray([[0, 0], [4, -3.5], [8, -6.5], [12, -10], [16, -12], [20, -11], [22, -9], [22, -5], [18, 0], [14, 3], [10, 7], [12, 11.5], [16, 11.5], [20, 8], [24.63, 3.08], [28, -1], [31, -1], [34, 3], [37, 6], [40, 9], [44, 13.5], [48, 17.5], [52, 22], [54, 25], [54.5, 29], [52.5, 33], [50, 37], [48, 41], [47.5, 43], [47.5, 45], [48.5, 49], [49, 52], [49, 54], [49.5, 58], [49, 60], [47, 61.5], [44, 62], [40, 62.5], [38, 63], [36, 66], [35, 70], [34, 74], [33, 78], [34, 80], [35, 82], [33, 85], [29.5, 89], [26, 93], [21, 97], [17, 99], [13, 98], [10.5, 94], [11.5, 90], [12.5, 86], [13.5, 82], [14, 78], [14, 76], [10.5, 72], [8, 68], [5.5, 64], [3.5, 60], [1.5, 56], [-0.5, 52], [-2.5, 48], [-4.5, 44], [-6.5, 40], [-7, 38], [-5.5, 34], [-3.5, 30], [-2, 26], [-3, 23], [-6, 21.5], [-10, 23], [-14.5, 27], [-19, 31], [-24, 35], [-28, 37], [-32, 38.5], [-36, 39.5], [-40, 39.5], [-44, 38.5], [44, 37], [-42.5, 35], [-38, 31], [-34, 28], [-30, 24.5], [-26, 21.5], [-20, 16.5], [-16, 13], [-12, 10], [-8, 6.5], [-4, 3]]) 
    return coordinates

def hockenheimLap():
    coordinates = np.asarray([[0, 0], [1, 0], [3, 1], [5, 2]]) #placeholder values
    return coordinates

def melbourneLap():
    # TBC
    coordinates = np.asarray([[0, 0], [1, 0], [3, 1], [5, 2]]) #placeholder values
    return coordinates

def saopaoloLap():
    # TBC
    coordinates = np.asarray([[0, 0], [1, 0], [3, 1], [5, 2]]) #placeholder values
    return coordinates

def shanghaiLap():
    # TBC
    coordinates = np.asarray([[0, 0], [1, 0], [3, 1], [5, 2]]) #placeholder values
    return coordinates

def testing():
    coordinates = np.asarray([[38.35, 4.7], [38.45, 5.8], [38.85, 6.8], [39.15, 7.8], [39.45, 8.8], [39.75, 9.8], [39.75, 10.9], [39.65, 11.9], [39.25, 12.9], [38.75, 13.8], [38.25, 14.7], [38.05, 15.7], [38.05, 16.8], [37.95, 17.9], [37.45, 18.8], [36.85, 19.7], [36.35, 20.6], [35.85, 21.5], [35.35, 22.4], [34.75, 23.3], [34.25, 24.2], [34.05, 25.2], [35.05, 25.5], [36.05, 25.3], [37.05, 25.1], [38.05, 24.6], [38.85, 23.8], [39.55, 23.0], [40.35, 22.2], [40.95, 21.3], [41.75, 20.5], [42.55, 19.7], [43.35, 18.9], [44.15, 18.1], [44.95, 17.3], [45.75, 16.5], [46.85, 16.5], [47.95, 16.5], [48.75, 17.3], [49.55, 18.1], [49.75, 19.1], [49.25, 20.1], [48.65, 21.0], [48.05, 21.9], [47.55, 22.8], [46.95, 23.6], [46.35, 24.5], [45.75, 25.3], [45.05, 26.2], [44.45, 27.0], [43.85, 27.8], [43.25, 28.6], [42.55, 29.5], [41.95, 30.3], [41.65, 31.3], [42.45, 32.1], [43.45, 32.5], [44.45, 32.6], [45.45, 32.7], [46.45, 32.8], [47.45, 32.9], [48.45, 33.0], [49.45, 33.2], [50.45, 33.3], [51.45, 33.4], [52.45, 33.5], [53.45, 33.6], [54.45, 33.8], [55.45, 33.9], [56.45, 34.0], [57.45, 34.1], [58.45, 34.3], [59.45, 34.4], [60.45, 34.5], [61.45, 34.6], [62.45, 34.8], [63.45, 34.9], [64.45, 35.0], [65.45, 35.1], [66.45, 35.3], [67.45, 35.4], [68.45, 35.6], [69.45, 35.7], [70.45, 35.8], [71.45, 36.0], [72.45, 36.2], [73.45, 36.3], [74.45, 36.5], [75.45, 36.7], [76.45, 36.8], [77.45, 37.0], [78.45, 37.2], [79.45, 37.4], [80.45, 37.5], [81.45, 37.7], [82.45, 37.9], [83.45, 38.1], [84.45, 38.3], [85.45, 38.5], [86.45, 38.7], [87.45, 38.9], [88.45, 39.1], [89.45, 39.4], [90.45, 39.6], [91.45, 39.8], [92.45, 40.0], [93.45, 40.3], [94.45, 40.5], [95.45, 40.7], [96.45, 41.0], [97.45, 41.2], [98.45, 41.5], [99.45, 41.7], [100.45, 42.0], [101.45, 42.2], [102.45, 42.5], [103.45, 42.7], [104.45, 43.0], [105.45, 43.3], [106.45, 43.6], [107.45, 43.8], [108.45, 44.1], [109.45, 44.4], [110.45, 44.7], [111.45, 45.0], [112.45, 45.3], [113.45, 45.6], [114.45, 45.9], [115.45, 46.2], [116.45, 46.5], [117.45, 46.8], [118.45, 47.1], [119.45, 47.5], [120.45, 47.8], [121.45, 48.1], [122.45, 48.4], [123.45, 48.8], [124.45, 49.1], [125.45, 49.4], [126.45, 49.8], [127.45, 50.1], [128.45, 50.4], [129.45, 50.5], [130.25, 49.8], [129.65, 48.9], [129.05, 48.0], [128.35, 47.1], [127.65, 46.2], [126.95, 45.4], [126.35, 44.6], [125.65, 43.7], [124.95, 42.9], [124.35, 42.1], [123.65, 41.3], [122.95, 40.4], [122.35, 39.6], [121.65, 38.8], [120.95, 37.9], [120.25, 37.1], [119.55, 36.2], [118.85, 35.3], [118.15, 34.5], [117.45, 33.6], [116.75, 32.7], [116.05, 31.8], [115.35, 31.0], [114.75, 30.2], [114.05, 29.4], [113.45, 28.6], [112.75, 27.8], [111.95, 27.0], [111.15, 26.2], [110.35, 25.4], [109.35, 25.0], [108.35, 24.8], [107.35, 24.6], [106.35, 24.4], [105.35, 24.1], [104.35, 23.9], [103.35, 23.7], [102.35, 23.5], [101.35, 23.3], [100.35, 23.0], [99.35, 22.8], [98.35, 22.7], [97.35, 23.1], [96.55, 23.9], [95.75, 24.7], [94.95, 25.5], [94.05, 26.1], [93.05, 26.4], [91.95, 26.4], [90.95, 26.2], [90.05, 25.7], [89.25, 24.9], [88.45, 24.1], [87.65, 23.3], [86.85, 22.5], [86.05, 21.7], [85.25, 20.9], [84.45, 20.1], [83.65, 19.4], [82.65, 19.0], [81.65, 18.9], [80.65, 19.2], [79.65, 19.6], [78.65, 20.1], [77.65, 20.5], [76.65, 21.0], [75.65, 21.4], [74.65, 21.9], [73.65, 22.2], [72.65, 22.4], [71.55, 22.4], [70.55, 22.3], [69.55, 22.0], [68.55, 21.5], [67.65, 21.0], [66.75, 20.4], [65.95, 19.6], [65.15, 18.8], [64.35, 18.0], [63.55, 17.2], [62.75, 16.4], [62.65, 15.4], [62.45, 14.4], [62.35, 13.4], [62.15, 12.4], [61.55, 11.5], [60.75, 10.7], [59.95, 9.9], [59.15, 9.1], [58.35, 8.3], [57.55, 7.7], [56.75, 7.0], [55.95, 6.2], [55.15, 5.4], [54.35, 4.6], [53.65, 3.8], [53.15, 2.8], [52.65, 1.9], [52.05, 1.1], [51.25, 0.5], [50.35, -0.1], [49.45, -0.6], [48.55, -1.2], [47.65, -1.8], [46.75, -2.3], [45.85, -2.9], [44.95, -3.5], [44.15, -4.3], [43.35, -5.1], [42.55, -5.9], [41.75, -6.7], [40.95, -7.5], [40.15, -8.3], [39.35, -9.1], [38.55, -9.9], [38.05, -10.8], [37.95, -11.8], [37.75, -12.8], [37.65, -13.8], [37.65, -14.9], [37.75, -15.9], [38.05, -16.9], [38.05, -18.0], [38.05, -19.1], [38.05, -20.2], [38.15, -21.3], [38.45, -22.3], [38.85, -23.3], [39.15, -24.3], [39.55, -25.3], [39.85, -26.3], [40.15, -27.3], [40.25, -28.3], [39.25, -28.2], [38.25, -27.8], [37.35, -27.3], [36.45, -26.8], [35.55, -26.2], [34.75, -25.6], [33.85, -24.9], [33.05, -24.3], [32.15, -23.6], [31.25, -22.9], [30.35, -22.2], [29.55, -21.6], [28.65, -20.9], [27.75, -20.2], [26.95, -19.6], [26.05, -18.9], [25.15, -18.2], [24.35, -17.6], [23.45, -16.9], [22.55, -16.3], [21.75, -15.6], [20.85, -15.0], [20.05, -14.3], [19.15, -13.7], [18.35, -13.0], [17.55, -12.4], [16.65, -11.7], [15.85, -11.1], [14.95, -10.4], [14.15, -9.8], [13.35, -9.2], [12.55, -8.6], [11.65, -7.9], [10.85, -7.3], [10.05, -6.7], [9.25, -6.1], [8.35, -5.4], [7.55, -4.8], [6.75, -4.2], [5.85, -3.5], [5.05, -2.9], [4.25, -2.3], [3.35, -1.6], [2.45, -0.9], [1.65, -0.3], [0.75, 0.4], [-0.15, 1.1], [-0.95, 1.7], [-1.75, 2.3], [-2.55, 2.9], [-3.45, 3.6], [-4.25, 4.2], [-5.25, 4.5], [-6.05, 5.3], [-6.85, 6.1], [-7.65, 6.9], [-7.45, 7.9], [-6.55, 8.5], [-5.55, 8.9], [-4.55, 9.4], [-3.55, 9.8], [-2.55, 10.3], [-1.55, 10.7], [-0.55, 11.2], [0.45, 11.6], [1.45, 12.1], [2.45, 12.5], [3.45, 13.0], [4.45, 13.4], [5.45, 13.9], [6.45, 14.3], [7.45, 14.8], [8.45, 15.2], [9.45, 15.6], [10.45, 15.9], [11.55, 15.9], [12.55, 15.6], [13.35, 14.9], [14.15, 14.1], [14.95, 13.3], [15.75, 12.5], [16.55, 11.7], [17.35, 10.9], [18.15, 10.1], [18.95, 9.3], [19.75, 8.5], [20.55, 7.7], [21.35, 6.9], [21.95, 6.0], [22.65, 5.1], [23.35, 4.2], [24.25, 3.5], [25.15, 3.0], [26.15, 2.7], [27.15, 2.4], [28.15, 2.1], [29.15, 1.9], [30.25, 1.9], [31.25, 1.8], [32.25, 2.1], [33.25, 2.5], [34.15, 3.0], [35.15, 3.4], [36.15, 3.9], [37.15, 4.2]])
    return coordinates