import numpy as np
from math import cos, sin, tan, radians, floor, ceil
from scipy.signal import find_peaks


class LineSegment:
    def __init__(self, pt_indices:np.ndarray, angle: float) -> None:
        self.pt_indices = pt_indices
        self.angle = angle
        self.slope = tan(radians(self.angle)) if self.angle != 90 else None

        self.x = None
        self.y = None

    @property
    def pt_indices(self):
        return self._pt_indices
    @pt_indices.setter
    def pt_indices(self, new_pts):
        self._pt_indices = new_pts

    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, new_x):
        self._x = new_x

    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, new_y):
        self._y = new_y

    @property
    def angle(self):
        return self._angle
    @angle.setter
    def angle(self, new_angle):
        self._angle = new_angle

    def average_pts(self, pt_list:np.ndarray):
        pt_coords = pt_list.take(self.pt_indices, axis= 0)
        cntr_pt = np.mean(pt_coords, axis= 0)
        self.x = cntr_pt[0]
        self.y = cntr_pt[1]

class RefinedOutline:
    def __init__(self, pts:np.ndarray, angle_interval: float = 1, 
    angle_gap: float = 30.0) -> None:

        #remove last point (AKA duplicate) from list
        self.pts = pts[:-1].copy()
        # 180 degree range split by provided angle interval
        self.angle_interval = angle_interval
        self.angle_range = round((180 + angle_interval) / angle_interval)
        self.all_angles = np.arange(0, self.angle_interval + 180, self.angle_interval)
        self.angle_gap = angle_gap

        #average distance between points in input outline
        self.outline_density = self._get_outline_density()
        #the two neccessary matrices
        self.r_matrix = self._r_matrix()
        self.ha_matrix = self._ha_matrix()

        #allow us to calculate...
        self.prom_div = 10 #prominence divisor for calculating dominant angles
        self._dominant_angles(self.prom_div) #calculate the dominant angles
        self.dom_vectors = self.np_angles_to_vect(self.dom_angles)
        self.top_angles = self.dom_angles[:2] #top two angles

        self.current_ix = 0 #pt to start from

        self.line_segments = []  #calculated line segments for intersection calculation

        self.intersection_pts = None #the corner points defining the refined polygon

        


    @property
    def pts(self):
        return self._pts
    @pts.setter
    def pts(self,new_pts):
        self._pts = new_pts

    @property
    def prom_div(self):
        return self._prom_div
    @prom_div.setter
    def prom_div(self,new_div):
        self._prom_div = new_div
    
    @property
    def dom_angles(self):
        return self._dom_angles
    @dom_angles.setter
    def dom_angles(self,new_ang):
        self._dom_angles = new_ang

    @property
    def intersection_pts(self):
        return self._intersection_pts
    @intersection_pts.setter
    def intersection_pts(self,new_pts):
        self._intersection_pts = new_pts

    def simple_calculate(self):
        fulcrum = self.pts[np.argmin(self.pts[:,1])]

        rotated_pts = self._rotate_points(self.pts, self.dom_angles[0] * -1, fulcrum)

        min_x, max_x = np.min(rotated_pts[:, 0]), np.max(rotated_pts[:, 0])
        min_y, max_y = np.min(rotated_pts[:, 1]), np.max(rotated_pts[:, 1])
        rotated_intersections = np.asarray([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

        self.intersection_pts = self._rotate_points(rotated_intersections, self.dom_angles[0], fulcrum)
        


    def calculate(self) -> None:

        #keeping track of the pt_indices
        previous_segment = self.line_segments[-1] if len(self.line_segments) else None

        num_angles = len(self.dom_angles)
        dist_tol = self.outline_density  # distance pts need to be to line

        pt = self.pts[self.current_ix]
        nbrs = self.pts[self.current_ix:]
        nbrs_ix = np.arange(self.current_ix,len(self.pts))

        #below matches as:
        #[a,b,c||a,b,c||a,b,c] = nbr pts
        #[x,x,x||x,x,x||x,x,x] = pt
        #[1,1,1||2,2,2||3,3,3] = direction vectors
        
        nbrs_rep = np.tile(nbrs, (num_angles, 1))
        pt_rep = np.tile(pt, (len(nbrs_rep), 1))
        dV_rep = np.repeat(self.dom_vectors, len(nbrs), axis= 0)

        #the lines to test against
        line_st, line_end = self._two_pt_line(pt_rep, dV_rep)

        #distance from nbr point to line
        distance = np.abs(np.cross(line_end - line_st, line_st - nbrs_rep)) \
                    / np.linalg.norm(line_end - line_st) * 100 
                    #Still unsure why I'm having to convert units / X100 ¯\_(ツ)_/¯

        dist_tol_mask = (distance <= dist_tol).astype(int)
        dist_tol_mask.shape = (num_angles, len(nbrs))

        #shifting the mask to left and right twice and summing to erase any gaps of 3 or less
        r_shift_1, r_shift_2 = np.roll(dist_tol_mask, 1, axis= 1), np.roll(dist_tol_mask, 2, axis= 1)
        l_shift_1, l_shift_2 = np.roll(dist_tol_mask, -1, axis= 1), np.roll(dist_tol_mask, -2, axis= 1)
        dist_tol_mask = ((r_shift_1 + r_shift_2 + l_shift_1 + l_shift_2 + dist_tol_mask) < 1).astype(int)

        #cumulatively summing neighbours within tolerance to get poitns directly adjacent from start pt
        proximity_level = np.cumsum(dist_tol_mask, axis=1)
        nbr_proximity_mask = (proximity_level == 0).astype(int)


        pts_per_line = np.sum(nbr_proximity_mask, axis= 1) #how many points captured by a line

        #grab the angle with the most points' tolerance mask and neighbours indices
        best_fit = np.argmax(pts_per_line)

        #move the pt list forward if the first few points are on a corner
        if len(self.line_segments) == 0 and (pts_per_line[best_fit] < len(self.pts) / 20 or best_fit > 1):
            self.pts = np.roll(self.pts, -1, axis= 0)
            self.calculate()
            return

        adjacent_nbrs = nbrs_ix[nbr_proximity_mask[best_fit, :].astype(bool)]

        if previous_segment and previous_segment.pt_indices[-1] + 1 == adjacent_nbrs[0] \
        and previous_segment.angle == self.dom_angles[best_fit]:
            previous_segment.pt_indices = np.concatenate((previous_segment.pt_indices, adjacent_nbrs))
        else:
            self.line_segments.append(LineSegment(adjacent_nbrs, self.dom_angles[best_fit]))

        self.current_ix = adjacent_nbrs[-1] + 1
        
        #main recursion
        if len(nbrs_ix) > 10 and self.current_ix < len(self.pts):
            self.calculate()
            return
        
        #merge the first and last if they are the same
        if self.line_segments[0].angle == self.line_segments[-1].angle:
            self.line_segments[0].pt_indices = \
            np.concatenate((self.line_segments[-1].pt_indices, self.line_segments[0].pt_indices))
            del self.line_segments[-1]


        #find the cntr pt of each line segment
        for seg in self.line_segments:
            seg.average_pts(self.pts)
        
        #first run
        self.intersection_pts = self._calc_intersections()


        # removing small segments
        delete_ix = []
        for ix, x_pt in enumerate(self.intersection_pts):
            l_segs= self.line_segments

            next_ix = self._wrap_index(ix + 1, 0, len(self.intersection_pts))
            nxt_pt = self.intersection_pts[next_ix]
            pt_2_pt_dist = np.linalg.norm(x_pt - nxt_pt)
            
            if pt_2_pt_dist < self.outline_density:
                if abs(l_segs[ix - 1].angle - l_segs[next_ix].angle) <= 5:
                    delete_ix.extend([ix, next_ix])
                else:
                    delete_ix.append(ix)
        
        for del_ix in delete_ix[::-1]:
            del l_segs[del_ix]
        
        # rerun it!
        #find the cntr pt of each line segment
        for seg in self.line_segments:
            seg.average_pts(self.pts)
        
        #first run
        self.intersection_pts = self._calc_intersections()
                    
    def _calc_intersections(self):
        intersections = []
        # see: https://math.stackexchange.com/q/1990698
        for i, line in enumerate(self.line_segments):
            nbr_line = self.line_segments[self._wrap_index(i + 1, 0, len(self.line_segments))]
            #test for horizontal vertical intersection
            if all(angle in [int(line.angle), int(nbr_line.angle)] for angle in [0, 90]):
                intersections.append(self._hor_vert_intx(line, nbr_line))
            elif 0 in [line.angle, nbr_line.angle]:
                print('horizontal / angle intersection')
                break
            elif 90 in [line.angle, nbr_line.angle]:
                print('vertical / angle intersection')
                break
            else:
                intersections.append(self._general_intx(line, nbr_line))

        return np.asarray(intersections)

    @staticmethod
    def _hor_vert_intx(og_line: LineSegment, nbr_line: LineSegment) -> list:
        line_angles = [og_line.angle, nbr_line.angle]
        hor_vert = [x for _, x in sorted(zip(line_angles, [og_line, nbr_line]))]
        return np.asarray([hor_vert[1].x, hor_vert[0].y])

    @staticmethod
    def _general_intx(l1: LineSegment, l2: LineSegment) -> list:
        intx_x = ((l2.slope * l2.x - l1.slope * l1.x) - (l2.y - l1.y)) / (l2.slope - l1.slope)
        intx_y = l1.slope * (intx_x - l1.x) + l1.y
        return np.asarray([intx_x, intx_y]) 

    def _find_pt_nbr_angles(self, ix: int):
            nbrs_ix = [ix - 1, ix + 1 if ix != len(self.line_segments) - 1 else 0]
            nbrs = [self.line_segments[nbrs_ix[0]], self.line_segments[nbrs_ix[1]]]
            return [nbrs[0].angle, nbrs[1].angle]

    @staticmethod
    def _wrap_index(index: int, start: int, end: int) -> int:
        return start + (index - start) % (end - start) if index >= 0 else end + index
   
    @staticmethod
    def _two_pt_line(points:np.ndarray, vectors:np.ndarray):
        line_start = points - vectors
        line_end = points + vectors
        return line_start, line_end


    def _get_outline_density(self):
        pt, nxt_pt = self.pts, np.roll(self.pts, 1, axis= 0)
        return np.mean(np.linalg.norm(pt - nxt_pt, axis= 1))


    def _r_matrix(self):
        """a matrix representing the r values as cell values, columns as angle intervals 
        between 0 and 180, and rows as indices of the input points.
        """
        r_angles_col = self.all_angles
        angle_matrix = np.tile(r_angles_col, (self.pts.shape[0], 1))

        x_val = np.tile(self.pts[:, 0],(angle_matrix.shape[1],1)).T
        y_val = np.tile(self.pts[:, 1],(angle_matrix.shape[1],1)).T

        return x_val * np.cos(np.deg2rad(angle_matrix)) + y_val * np.sin(np.deg2rad(angle_matrix))

    def _angle_bins(self):
        bin_min = floor(self.r_matrix.min())
        bin_max = ceil(self.r_matrix.max())
        bins = np.arange(bin_min, bin_max, self.outline_density)
        indices_binned = np.digitize(self.r_matrix, bins, right= False)
        
        return len(bins), indices_binned


    def _ha_matrix(self):
        bin_num, indices_binned = self._angle_bins()

        matrix_ha = np.zeros((bin_num, self.angle_range), dtype= int)
        
        for row in indices_binned:
            for j, cell in enumerate(row):
                matrix_ha[cell - 1, j] = matrix_ha[cell - 1, j] + 1
        
        return matrix_ha

    def _dominant_angles(self, prominence_divisor:float):
        ha_std_dev = np.std(self.ha_matrix, axis = 0)
        #shifting the filtered values to use find peaks on 0 degree line (horizontal)
        shifted_std_dev = np.roll(ha_std_dev, 90) 

        p_prominence = np.median(ha_std_dev) / prominence_divisor

        peaks = find_peaks(ha_std_dev, prominence= p_prominence, distance= self.angle_gap)[0]
        shifted_peaks = find_peaks(shifted_std_dev, prominence= p_prominence, distance= self.angle_gap)[0]
        
        #inserting horizontal direction indices (if any)
        if any(x in shifted_peaks for x in [89, 89.5, 90, 90.5, 91]):
            peaks = np.insert(peaks, 0, -1)

        #if we don't have atleast 2 angles we need to decrease the required prominence
        if len(peaks) <= 1:
            self.prom_div += 1
            self._dominant_angles(self.prom_div)
        else:
            #sorting the angles by frequency
            dom_std_dev = np.take(ha_std_dev, peaks)
            peaks = np.asarray([pk for _, pk in sorted(zip(dom_std_dev, peaks))][::-1])

            #back to 0->180
            peaks = peaks * self.angle_interval
            neg_check = np.vectorize(lambda x: x + 180 if x < 0 else x)
            self.dom_angles = neg_check(peaks)

    @staticmethod
    def angle_to_vector(angle: float) -> float:
        rad = radians(angle)
        return np.asarray([cos(rad) * 10000, sin(rad) * 10000])#big numbers for long test line

    @staticmethod
    def np_angles_to_vect(angles: np.ndarray) -> np.ndarray:
        rad = np.deg2rad(angles)
        vec_x = np.cos(rad) * 1000
        vec_y = np.sin(rad) * 1000
        return np.stack((vec_x, vec_y)).T

    @staticmethod
    def x_closest_vals(in_arr: np.ndarray, val: float, x: int = 1):
        if x == 1:
            return in_arr[(np.abs(in_arr - val)).argmin()]

        search_arr = in_arr.copy()
        ix_list = []
        for _ in range(x):
            ix = (np.abs(search_arr - val)).argmin()
            search_arr[ix] = 666777888999 #set the value to something that won't be picked
            ix_list.append(ix)
        
        return ix_list

    @staticmethod
    def _rotate_points(pts:np.ndarray, angle: float, origin:np.ndarray) -> np.ndarray:
        
        rad = np.deg2rad(angle)
        ox, oy = origin[0], origin[1]
        px, py = pts[:, 0], pts[:, 1]

        qx = ox + np.cos(rad) * (px - ox) - np.sin(rad) * (py - oy)
        qy = oy + np.sin(rad) * (px - ox) + np.cos(rad) * (py - oy)

        return np.column_stack((qx, qy))


    