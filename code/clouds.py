from math import floor, ceil
import warnings
import numpy as np
import laspy
from sklearn.cluster import DBSCAN

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


class SiteCloud():
    def __init__(self, las_path: str) -> None:
        """
        A LIDAR las file converted to a 3xN point array. Attributes contain ASPR classified clouds.

        Init Args
        ----------
        las_path (str): 
         The absolute or relative path of a valid las LIDAR file.

        Attributes
        ----------
        las_path (str): 
         The absolute path of a valid las LIDAR file.
        
        cld_name (str): 
         The SiteCloud instance name. Defaults to las file name.

        pt_array (ndarray): 
         All points in the las file.

        translation_vector (ndarray): 
         The 3D vector from the min X,Y,Z coordinates of the cloud to 0,0,0. 

        available_classes (list[str]): 
         The ASPRS classes found in the las file. 
         Ref: http://www.asprs.org/a/society/committees/lidar/LAS_1-4_R6.pdf page 10

        ***Dyanmic Attribute(s)*** ASPR classified point clouds (ndarray): 
         The SiteCloud class instance will generate subclouds for each found ASPR class.
        Attribute 'available_classes' will list all generated sub-cloud attributes.

        Methods
        -------
        translate_cloud(self) -> None: 
         Move all clouds to 0,0,0 if clouds are at original position. Move all clouds to original position 
         if they have already been translated.

        cluster_in_plan(self, classif: str, seperation: float = 1) -> list[np.ndarray]:
         Clusters chosen ASPR class sub-cloud in plan. Useful for seperating buildings, trees, etc.
        """
        
        self.las_path = os.path.abspath(las_path)

        if not (os.path.isfile(self.las_path) and self.las_path.endswith('.las')):
            raise ValueError('You did not provide a valid file path or las file.')
    
        self.cld_name = self.__site_name()

        self.__cld_classifications, self.pt_array, self.translation_vector = self.__las_pt_array()

        self.__translated = False #toggle for whether or not cloud is moved to origin

        self.available_classifs = self.__aspr_list() #ASPR classifications present in the main cloud


    #-------------------SETTERS-------------------------------------------------------------------------------

    @property
    def cld_name(self):
        return self._cld_name
    @cld_name.setter
    def cld_name(self, new_name):
        self._cld_name = new_name

    @property
    def pt_array(self):
        return self._pt_array
    @pt_array.setter
    def pt_array(self, new_pts):
        self._pt_array = new_pts
    
    @property
    def translated(self):
        return self._translated
    @translated.setter
    def translated(self, new_val):
        self._translated = new_val
    
    #-------------PRIVATE-------------------------------------------------------------------------------------
    def __site_name(self) -> str:
        """Sets cloud instance name based on LAS file name."""

        file_name = os.path.basename(self.las_path)
        return os.path.splitext(file_name)[0]
    
    def __las_pt_array(self) -> 'tuple(np.ndarray)':
        """Returns pt array, classification array, and min XYZ for the given las file."""

        laspy_cld = laspy.read(self.las_path)

        pt_class = laspy_cld.classification

        #divide by 100 to maintain meter units
        pt_array = np.stack([laspy_cld.X /100, laspy_cld.Y /100,\
             laspy_cld.Z /100], axis= 0).transpose((1,0)) 

        min_x = np.min(pt_array[:, 0])
        min_y = np.min(pt_array[:, 1])
        min_z = np.min(pt_array[:, 2])

        return pt_class, pt_array, np.asarray([[min_x, min_y, min_z]])


    def __aspr_cloud(self, aspr_classif: int) -> np.ndarray:
        """Returns pt array containing only the provided ASPRS classification."""

        aspr_mask = self.__cld_classifications == aspr_classif
        out_pts = self.pt_array[aspr_mask]

        return None if out_pts.shape[0] == 0 else out_pts


    def __aspr_list(self) -> 'list[str]':
        """Returns a list of available ASPRS classed sub clouds and sets attributes with values of
        the corresponding filter point cloud. \n 
        ref: http://www.asprs.org/a/society/committees/lidar/LAS_1-4_R6.pdf page 10"""

        ####NEED TO EVENTUALLY CLASSIFY USER DEFINED PTS - SEE REF ABOVE FOR NUMBERS.

        classif_list = ['created_never_classified', 'unclassified', 'ground', 'low_veg', 'med_veg', 
        'high_veg', 'buildings', 'low_noise', 'high_noise', 'water', 'rails', 'roads', 'bridges', 
        'wire_gaurds', 'wire_con', 'trans_twr', 'wire_struct']

        avail_classifs = []
        for ix, classification in enumerate(classif_list):
            classif_pts = self.__aspr_cloud(ix)
            if classif_pts is not None:
                avail_classifs.append(classification)
                self.__setattr__(classification, classif_pts)

        return avail_classifs

    #----------PUBLIC-----------------------------------------------------------------------------------------

    def translate_cloud(self):
        """ Move all clouds to 0,0,0 if clouds are at original position. \n
        Move all clouds to original position if they have already been translated.
        """
        trans_vec = np.negative(self.translation_vector) if self.__translated else self.translation_vector
        self.__translated = not self.__translated

        self.pt_array = np.subtract(self.pt_array, trans_vec)

        #moving all the classified sub-clouds as well
        for classif in self.available_classifs:
            self.__setattr__(classif, np.subtract(self.__getattribute__(classif), trans_vec))

    def cluster_in_plan(self, classif: str, seperation: float = 1) -> 'list[np.ndarray]':
        """Clusters chosen ASPR class sub-cloud in plan. Useful for seperating buildings, trees, etc.

        Args:
            classif (str): The cloud to be clustered. Check SiteCloud.available_classifs for classes in 
            current cloud.

            seperation (float, optional): The required seperation distance between clusters. Defaults to 1.

        Returns:
            list[np.ndarray]: A list of point arrays corresponding to the clustered clouds.
        """
        cld_to_cluster = self.__getattribute__(classif)

        db = DBSCAN(eps= seperation, min_samples= 10).fit(cld_to_cluster[:,:2])
        db_lbls = [lbl for lbl in np.unique(db.labels_) if lbl != -1]
        
        cloud_clusters =[]
        for lbl in db_lbls:
            mask = db.labels_ == lbl
            cloud_clusters.append(self.__getattribute__(classif)[mask])
        
        return cloud_clusters


#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#neighbourhood classifcation list for voxelization


good_nbrhds = [
     #insulated 
    # [*][*][*]
    # [*][X][*]
    # [*][*][*]
    111111111, 111101111,

    #chipped
    # [*][*][ ]
    # [*][X][*]
    # [*][*][*]
    110111111, 111111110, 111111011, 11111111, 

    #dinged (+ flipped horizontally)
    # [*][ ][ ]
    # [*][X][*]
    # [*][*][*]
    100111111, 111110110, 111111001, 11011111, 1111111, 110110111, 111111100, 111011011,   

    #sliced
    # [ ][ ][ ]
    # [*][X][*]
    # [*][*][*]
    111111, 110110110, 111111000, 11011011,

    #chamfered
    # [ ][ ][ ]
    # [*][X][ ]
    # [*][*][ ]
    110110, 110110000, 11011000, 11011,

    #flagged (+ flipped horizontal)
    # [*][ ][ ]
    # [*][X][ ]
    # [*][*][ ]
    100110110, 111110000, 11011001, 11111, 1011011, 110111, 110110100, 111011000,

    #beaked
    # [*][ ][ ]
    # [*][X][ ]
    # [*][*][*]
    100110111, 111110100, 111011001, 1011111,

    #pitted
    # [*][ ][*]
    # [*][X][*]
    # [*][*][*]
    101111111, 111110111, 111111101, 111011111

    #stretched
    # [ ][*][*]
    # [*][X][*]
    # [*][*][ ]
    #11111110, 110111011, 11111110, 110111011
    #maybe? could lead to intersecting edges

]

class BldgCloud():
    def __init__(self, pt_array:np.ndarray, initial_translation:np.ndarray = np.array([0,0,0]), 
    angle_interval: float = 0.5, voxel_interval:float = 1.0) -> None:

        self.pt_array = self.__pt_array_init(pt_array) #input cloud pt array
        self.in_translation = initial_translation # in case we need to move the cloud back

        self.__angle_int = angle_interval #angle to rotate caliper bounding box

        self.__vox_int = voxel_interval #the density of the voxels

        self.dom_angle, self.__box_crnrs, self.__rot_points = self.__initial_dom_directions()

        self.histogram, self.voxel_cntrs = self.__voxelize_pts()

        self.initial_pt_bool = self.__clean_points(self.histogram, good_nbrhds) #all the lidar points
        self.unique_levels = self.__remove_repeat_lvls(self.initial_pt_bool)
        self.level_outlines = self.__level_outlines(self.initial_pt_bool, self.unique_levels)
        self.bottom_pt = self.__first_point(self.level_outlines)



    #-------------SETTERS-------------------------------------------------------------------------------------

    #-------------PRIVATE-------------------------------------------------------------------------------------

    @staticmethod
    def __pt_array_init(pt_array: np.ndarray) -> np.ndarray:
        """Checks if pt_array is valid"""

        if not isinstance(pt_array, np.ndarray) or len(pt_array.shape) != 2:
            raise ValueError('Please provide a valid N x 3 point ndarray')
        elif pt_array.shape[1] == 3:
            return pt_array
        elif pt_array.shape[1] == 2:
            warnings.warn('Only a 2D cloud was provided. Filled z-axis with zeros.')
            z_vals = np.reshape(np.zeros_like(pt_array[:,0]), (-1, 1))
            return np.concatenate((pt_array, z_vals), axis= 1)
        else:
            raise ValueError('Please provide a valid N x 3 point ndarray')


    def __initial_dom_directions(self):
        """Returns dominant angle for a given cloud and angle aligned bounding box corner points. 
        Loosely based on rotating caliper algorithm."""

        best_angle = None
        best_corners = None
        best_area = 987654321987654321987654321 #giant place holder number

        for i in self.__angle_range():
            angle = i * self.__angle_int
            rotated_pts = self.__rotate_points(self.pt_array, angle)
            area, corners = self.__b_box_area(rotated_pts)

            if area < best_area:
                best_area = area
                best_angle = angle
                best_corners = corners
        
        best_corners.extend([np.min(self.pt_array[:,2]), np.max(self.pt_array[:,2])])

        return best_angle, best_corners, self.__rotate_points(self.pt_array, best_angle)

    def __angle_range(self):
        """Returns range of angles for the given angle interval"""
        angle_range = round((180) / self.__angle_int)
        return range(angle_range)


    @staticmethod
    def __b_box_area(pts: np.ndarray) -> tuple():
        """Generates a in-plan bounding box for a point array and returns box area."""
        min_x, max_x = np.min(pts[:,0]), np.max(pts[:,0])
        min_y, max_y = np.min(pts[:,1]), np.max(pts[:,1])
        return abs(max_x - min_x) * abs(max_y - min_y), [min_x, max_x, min_y, max_y]


    @staticmethod
    def __rotate_points(pts:np.ndarray, angle: float, origin:np.ndarray = np.array([0,0,0])) -> np.ndarray:
        """Rotates an array of points around an origin which defaults to 0,0,0."""
        
        rad = np.deg2rad(angle)
        ox, oy = origin[0], origin[1]
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

        qx = ox + np.cos(rad) * (px - ox) - np.sin(rad) * (py - oy)
        qy = oy + np.sin(rad) * (px - ox) + np.cos(rad) * (py - oy)

        return np.column_stack((qx, qy, pz))


    def __voxelize_pts(self) -> tuple:
        """Generates voxel cntr points as well as the number of cloud points captured in that voxel."""

        rot_pts = self.__rot_points
        x_bins, y_bins, z_bins = self.__setup_bins('x'), self.__setup_bins('y'), self.__setup_bins('z')

        #bins is bins -1 since we are defining the number of spaces between the bin edges
        #histo represents the number of points that have fallen in each bin
        histo, edges = np.histogramdd(sample= (rot_pts[:, 0], rot_pts[:, 1], rot_pts[:, 2]),
                        bins= (len(x_bins) - 1, len(y_bins) - 1, len(z_bins) - 1), 
                        range= ((x_bins[0], x_bins[-1]),(y_bins[0], y_bins[-1]),(z_bins[0], z_bins[-1])))

        #converting the bin edges to voxel cntr points
        voxel_ctrs = [b_edges[:-1] + self.__vox_int / 2 for b_edges in edges]

        filled_histo = self.__fill_vox_below(histo) #filling all areas below top most positive voxel

        transposed_histo = np.transpose(filled_histo, (2,0,1)) #transpose for Z,X,Y format (grouped by level)

        return transposed_histo, voxel_ctrs


    def __setup_bins(self, dimension: str):
        """A range for the voxel bins. Needs x,y,x to be specified for dimension."""
        dim = {'x':0, 'y':2, 'z':4}
        min_val = self.__box_crnrs[dim[dimension]]
        max_val = self.__box_crnrs[dim[dimension] + 1]

        r_min = min_val - (min_val % self.__vox_int) - self.__vox_int
        r_max = max_val - (max_val % self.__vox_int) + 2 * self.__vox_int + self.__vox_int
        #in the above I added a 2 voxel padding. Required later for the lookup table.

        return np.arange(r_min, r_max, self.__vox_int)


    @staticmethod
    def __fill_vox_below(histo:np.ndarray) -> np.ndarray:
        """Input the histogram and fill all voxels below highest positive values with a value. 
        Histogram is formatted as an array with shape (Z dimensions, X dimensions, Y dimensions)"""

        for x in range(histo.shape[0]):
            for y in range(histo.shape[1]):
                column = histo[x, y]
                if np.sum(column) != 0:
                    #returns the index above the last aka (highest in elevation) non zero value
                    top_ix_bounds = histo.shape[2] - np.flip(column != 0).argmax()
                    histo[x, y, :top_ix_bounds] = 1
        
        return histo

    @staticmethod
    def __density_nbrhd(pts_in: np.ndarray) -> np.ndarray:
        """Calculates the relationship of nbrs for each cell, given by a 8 or 9 digit number where each
           digit refers to a neighbour as shown below
        
            | ul_nbr       | u_nbr        | ur_nbr       |
            | *100,000,000 | *10,000,000  | *1,000,000   |
            ----------------------------------------------
            | l_nbr        | nbrhd_ctr    |  r_nbr       |
            | *100,000     | *10,000      | *1,000       |
            ----------------------------------------------
            | dl_nbr       | d_nbr        | dr_nbr       |
            | *100         | *10          | *1           |
        """

        ul_nbr = np.roll(pts_in, 1, axis=(1,2)) * 100000000
        u_nbr = np.roll(pts_in, 1, axis=1) * 10000000
        ur_nbr = np.roll(pts_in, (1, -1), axis=(1,2)) * 1000000

        l_nbr = np.roll(pts_in, 1, axis=2) *100000
        nbrhd_cntr = pts_in * 10000
        r_nbr = np.roll(pts_in, -1, axis=2) * 1000

        dl_nbr = np.roll(pts_in, (-1, 1), axis=(1,2)) * 100
        d_nbr = np.roll(pts_in, -1, axis=1) * 10
        dr_nbr = np.roll(pts_in, -1, axis=(1,2))

        return ul_nbr + u_nbr + ur_nbr + l_nbr + nbrhd_cntr + r_nbr + dl_nbr + d_nbr + dr_nbr

    def __clean_points(self, pts_in: np.ndarray, nbhrhd_list: list) -> np.ndarray:
        """ Returns a 0/1 mask of points who meet neighbourhood requirement."""
        initial_nbhrhds = np.isin(self.__density_nbrhd(pts_in), nbhrhd_list)
        
        #fill small holes before removing points and convert to int type
        initial_nbhrhds = (initial_nbhrhds.astype(int) + pts_in > 0).astype(int)

        return np.isin(self.__density_nbrhd(initial_nbhrhds), nbhrhd_list).astype(int)


    @staticmethod
    def __remove_repeat_lvls(pts: np.ndarray, pt_tolerance: int = 5):
        """Returnslevel numbers corresponding to the unique (enough) levels of the original point array. """
        prev_lvl, unique_lvls = 0, [0]
        
        for ix, current_lvl in enumerate(pts[1:]):
            if np.abs(np.sum(current_lvl - pts[prev_lvl])) >= pt_tolerance:
                prev_lvl = ix + 1
                unique_lvls.append(ix + 1)

        return unique_lvls


    def __level_outlines(self, lvl_pts:np.ndarray, unique_lvls: list) ->np.ndarray:
        """Returns only the outline points for the unique levels of a given voxelized cloud. Also removes
        T-junction type points that look like: 
        [ ][ ][ ]
        [ ][X][ ]
        [*][*][*]"""

        #t_junctions = [100110100, 10111, 1011001, 111010000]
        insulated = [111111111, 111101111]
        #bad_nbrhds = t_junctions + insulated


        unique_pts = lvl_pts[unique_lvls, :, :]
        interior_spaces = np.isin(self.__density_nbrhd(unique_pts), insulated).astype(int)
        return (unique_pts - interior_spaces > 0).astype(int)

    def order_pts(self, pts: np.ndarray) ->np.ndarray:
        """returns an array of ordered pt indices"""
        corner_points = []
        first_pts = self.__first_point(pts)
        adjacencies = self.__adjacency_nbrhd(pts)

        for ix, level in enumerate(adjacencies):
          crnr_pts = self.__order_level(level, first_pts[ix])
          corner_points.append(crnr_pts)
        
        return(corner_points)
            

    def __order_level(self, neighborhoods: np.ndarray, first_pt: int):
        """returns an ordered list of indices for given neighbourhood relations"""

        #the nbrhrds representing an ortho line or diagonal line
        redundant_nbrhds = [10100000, 1010000, 1010, 101, 10101100, 1010110, 10100011, 1011001]

        column_count = self.level_outlines.shape[2]
        pt_list = [[first_pt]]

        nbrhds = np.copy(neighborhoods)
        nbrhd_list= [[neighborhoods[first_pt]]]

        for _ in range(20000):
            current_ix = pt_list[-1][-1] #grab the ix
            current_nbrhd = nbrhds[current_ix] #the current nbrhd condition
            ix_step, nbr_subtact = self.__ix_advance(current_nbrhd, column_count)
            next_ix = current_ix + ix_step

            if nbrhds[current_ix] == 0 and np.sum(nbrhds) == 0:
                break

            elif nbrhds[current_ix] == 0:
                new_loop_ix = self.__first_point(nbrhds)
                pt_list.append([new_loop_ix])
                nbrhd_list.append([neighborhoods[new_loop_ix]])

            elif nbrhds[next_ix] == 0:
                nbrhds[current_ix] = self.__remove_highest_nbr(current_nbrhd)

            else:
                nbrhds[current_ix] = 0
                nbrhds[next_ix] = nbrhds[next_ix] - nbr_subtact
                pt_list[-1].append(next_ix)
                nbrhd_list[-1].append(neighborhoods[next_ix])

        corner_pts = []
        for loop_ix, loop in enumerate(pt_list):
            if len(loop) > 2:
                corner_pts.append([])
                for item_ix, item in enumerate(loop):
                    if nbrhd_list[loop_ix][item_ix] not in redundant_nbrhds:
                        corner_pts[-1].append(item)
        
        return corner_pts



            

    @staticmethod
    def __remove_highest_nbr(nbrhd: int) -> int:
        """Removes the highest valued neighbour from neighbourhood (if there are any neighbours)"""
        if nbrhd == 0:
            return 0
        biggest_digit = 10 ** (floor(log10(nbrhd)))
        return nbrhd - biggest_digit

    @staticmethod
    def __ix_advance(nbrhd: int, col_count: int):
        """given current a current nbhrd adjacency patten (see __adjacency_nbrhd) located at point A, 
        return the next point in the outline (point B) by choosing the highest value neighbour starting
        from the left and rotating clockwise. The second returned value removes point A from point B's
        available neighbours by subtracting a value equal to point A using the __adjacency_nbrhd scheme."""
        
        if nbrhd >= 10000000: # point B->A = left
            return col_count * -1, \
            100000 # point A->B = right

        elif nbrhd >= 1000000: # point B->A = upper
            return 1, \
                10000 # point A->B = lower

        elif nbrhd >= 100000: # point B->A = right
            return col_count, \
                10000000 # point A->B = left
        
        elif nbrhd >= 10000: # point B->A = lower
            return -1, \
                1000000 # point A->B = upper

        elif nbrhd >= 1000: # point B->A = upper left
            return col_count * -1 + 1, \
                10 # point A->B = lower right

        elif nbrhd >= 100: # point B->A = upper right
            return col_count + 1, \
                1 # point A->B = lower left

        elif nbrhd >= 10: # point B->A = lower right
            return col_count - 1, \
                1000 # point A->B = upper left

        elif nbrhd == 1: # point B->A = lower left
            return col_count * -1 - 1, \
                100 # point A->B = upper right
        
        else: # no neighbours!
            return 0 ,0


    @staticmethod
    def __adjacency_nbrhd(pts_in: np.ndarray) -> np.ndarray:
        """Returns an an 2D array (XY flattened) of neighbourhood adjacency values with the scheme 
            | ul_nbr       | u_nbr        | ur_nbr       |
            | *1,000       | *1,000,000   | *100         |
            ----------------------------------------------
            | l_nbr        | nbrhd_ctr    |  r_nbr       |
            | *10,000,000  | multiplier   | *100,000     |
            ----------------------------------------------
            | dl_nbr       | d_nbr        | dr_nbr       |
            | *1           | *10,000      | *10          |
            note - I rotated this 90 degrees versus the __density neighbourhood. Might want to revisit
            the density neighbourhood even though rotations don't matter. Just to keep it consistent.
            further note - we also need to prioritize the cardnial direction before dealing with diagonals"""

        #ortho directions
        l_nbr = np.roll(pts_in, 1, axis=1) * 10000000
        u_nbr = np.roll(pts_in, -1, axis=2) * 1000000
        r_nbr = np.roll(pts_in, -1, axis=1) * 100000
        d_nbr = np.roll(pts_in, 1, axis=2) * 10000

        #diagonals
        ul_nbr = np.roll(pts_in, (1, -1), axis=(1,2)) * 1000
        ur_nbr = np.roll(pts_in, -1, axis=(1,2)) * 100
        dr_nbr = np.roll(pts_in, (-1, 1), axis=(1,2)) * 10
        dl_nbr = np.roll(pts_in, 1, axis=(1,2))
        
        nbr_arr = (l_nbr + ul_nbr + u_nbr + ur_nbr + r_nbr + dr_nbr + d_nbr + dl_nbr) * pts_in #empty cells will be 0
        return nbr_arr.reshape((nbr_arr.shape[0], nbr_arr.shape[1] * nbr_arr.shape[2]))

    @staticmethod
    def __first_point(arr: np.ndarray):
        """returns the index of the first non-zero point per each level"""
        if arr.ndim != 3:
            return arr.argmax()

        flat_lvls = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
        return flat_lvls.argmax(axis=1)
