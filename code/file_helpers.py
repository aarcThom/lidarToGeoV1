
# FILE HELPERS ARE FUNCTIONS USED SPECIFICALLY TO TRANSLATE BETWEEN FILE FORMATS

import numpy as np

def point_list(file_path: str):
    """Used to convert per line point coordinates 
    in format X,Y,Z to numpy array, minimum elevation, and XY translation

    Args:
        file_path (str): Coordinates file path.

    Returns:
        numpy.ndarray: 2D array of points X,Y coordinates
        float: The minimum elevation (z-coord.) of all the points
        numpy.ndarray: X,Y translation vector
    """
    point_list = []
    elevation_list = []

    with open(file_path) as pt_file:
        for line in pt_file:
                for char in [' ', ';', ',']:
                    if char in line:
                        coordinates = line.split(char)
                        elevation_list.append(float(coordinates[2]))
                        point_list.append([float(coordinates[0]), float(coordinates[1])])
                        break
    
    point_array = np.array(point_list)
    translation_vector = np.min(point_array, axis=0)

    point_array = point_array - translation_vector

    return point_array, min(elevation_list), translation_vector
