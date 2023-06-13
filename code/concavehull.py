import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import Point


class ConcaveHull(object):
    """ A Concave hull class: A k-nearest neighbours approach for the computation of the region occupied by a set of points (Moreira, Santos 2007).

    Args:
        points (np.ndarray or list of points): The points to wrap a concave hull around
        prime_index (int): Defaults to 0. Index for a list of primes. 0 = 3, 1 = 5, 2 = 7, etc. Override to consider more neighbours out of the gate.

    """

    def __init__(self, points, prime_index=0):

        if isinstance(points, np.core.ndarray):
            self.data_set = points
        elif isinstance(points, list):
            self.data_set = np.array(points)
        else:
            raise ValueError('please provide an [N,2] numpy array or a list of lists.')

        # clean up duplicates
        self.data_set = np.unique(self.data_set, axis=0)

        # create a spatial index used to filter already used points
        self.indices = np.ones(self.data_set.shape[0], dtype=bool)

        # prime numbers to use for k-nearest neighbours.
        self.prime_k = np.array(
            [3, 5, 7, 11, 13, 17, 21, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])

        self.prime_index = prime_index

    @staticmethod
    def find_min_y_index(pts):
        """
        grabs the point with the minimum y-value
        :param pts: points to check for min y
        :return: index of point with lowest y-value
        """
        indices = np.argsort(pts[:, 1])
        return indices[0]

    def get_next_k(self):
        """
        use next prime number as # of neighbours to search
        :return: k value
        """
        if self.prime_index < len(self.prime_k):
            return self.prime_k[self.prime_index]
        else:
            return -1

    def get_k_nearest(self, ix, k):
        """
        Calculates the k nearest point indices to the point indexed by ix
        :param ix: Index of starting point
        :param k: number of neighbours to consider
        :return: Array of indices into the data set array
        """
        ixs = self.indices
        base_indices = np.arange(len(ixs))[ixs]

        # see https://stackoverflow.com/a/50110565 for calculating the distance from one point to many
        distances = np.linalg.norm(self.data_set[ix, :] - self.data_set[ixs, :], axis=1)
        sorted_indices = np.argsort(distances)

        kk = min(k, len(sorted_indices))
        k_nearest = sorted_indices[range(kk)]
        return base_indices[k_nearest]

    def sort_by_angle(self, origin_index, nbr_indices, prev_index, count):
        """
        Returns indices of k nearest points - sorted by right-hand angle from previous line's angle
        :param origin_index: index of point to measure from
        :param nbr_indices: indices of neighbours to measure angles
        :param prev_index: index of previous origin point
        :param count: count from main loop
        :return: indices of neighbours sorted by angle
        """
        orig_x = self.data_set[origin_index, 0]
        orig_y = self.data_set[origin_index, 1]

        nbrs_x = self.data_set[nbr_indices, 0]
        nbrs_y = self.data_set[nbr_indices, 1]

        if count == 0:
            prev_x = orig_x + 10  # set first angle to 0 with point directly right of first origin
            prev_y = orig_y
        else:
            prev_x = self.data_set[prev_index, 0]
            prev_y = self.data_set[prev_index, 1]

        prev_angle = np.arctan2(prev_y - orig_y, prev_x - orig_x)
        current_angles = np.arctan2(nbrs_y - orig_y, nbrs_x - orig_x)

        angles = current_angles - prev_angle
        angles = np.rad2deg(angles)
        angles = np.mod(angles + 360, 360)

        return np.argsort(-angles)

    def recurse_calculate(self):
        """
        calculates the concave hull using the next value for k while reusing the distances dictionary
        :return: concave hull
        """
        recurse = ConcaveHull(self.data_set, self.prime_index + 1)
        next_k = recurse.get_next_k()
        return None if next_k == -1 else recurse.calculate(next_k)

    def calculate(self, k=3):
        """
        Calculates the convex hull of the data set as an array of points
        :param k: Number of nearest neighbours
        :return: Array of points (N,2) with the concave hull of a dataset
        """
        # less than 3 points doesn't work. 3 points is the concave hull
        if self.data_set.shape[0] < 3:
            return None
        if self.data_set.shape[0] == 3:
            return self.data_set

        # making sure that k neighbours can be found
        kk = min(k, self.data_set.shape[0])

        first_point = self.find_min_y_index(self.data_set)
        current_point = first_point

        # note that the hull and test_hull are matrices (N,2)
        hull = np.reshape(np.array(self.data_set[first_point, :]),
                          (1, 2))  # converts the hull from a 1D array to a 2D array with 1 row
        test_hull = hull

        # remove the first point
        self.indices[first_point] = False

        step = 0
        stop = kk
        prev_point = 0

        while ((current_point != first_point) or (step == 0)) and len(self.indices[self.indices]) > 0:
            if step == stop:
                self.indices[first_point] = True  # we can re-include the first point when searching for neighbours

            # get k nearest neighbours' indices
            k_nbrs = self.get_k_nearest(current_point, kk)

            # sort indices of k nearest neighbours by angle (descending, right hand turn)
            candidates = self.sort_by_angle(current_point, k_nbrs, prev_point, step)

            i = 0
            invalid_hull = True

            while invalid_hull and i < len(candidates):
                candidate = candidates[i]

                # creating a test hull to check if there are self-intersections
                next_point = np.reshape(self.data_set[k_nbrs[candidate]], (1, 2))
                test_hull = np.append(hull, next_point, axis=0)

                # using shapely to draw a line
                line = LineString(test_hull)
                invalid_hull = not line.is_simple  # shapely line is simple if it doesn't self-intersect
                i += 1

            if invalid_hull:  # recursing with the next prime number if no valid hull found
                return self.recurse_calculate()

            prev_point = current_point
            current_point = k_nbrs[candidate]
            hull = test_hull

            self.indices[current_point] = False
            step += 1

        poly = Polygon(hull)

        count = 0
        total = self.data_set.shape[0]
        for i in range(total):
            pt = Point(self.data_set[i, :])
            if poly.intersects(pt) or pt.within(poly):
                count += 1
            else:
                d = poly.distance(pt)
                if d < 0.001:
                    count += 1

        if count == total:
            return hull
        else:
            return self.recurse_calculate()