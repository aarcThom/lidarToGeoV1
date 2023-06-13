import os.path
from random import random
from math import fabs

import laspy
import open3d as o3d
import numpy as np
from scipy.signal import savgol_filter
from code.concavehull import ConcaveHull
from code.houghoutline import RefinedOutline

from shapely.geometry import Polygon
from trimesh import creation


#---------------class lookup tables-----------------------------------------------------

_asprs_classes = {0: 'created never classified', 1: 'unclassfied', 2: 'ground', 
3: 'low vegetation', 4: 'medium vegetation', 5: 'high vegetation', 6: 'buildings',
7: 'low noise', 8: 'high noise', 9: 'water', 10: 'rails', 11: 'road surfaces',
12: 'bridge decks', 13: 'wire guards', 14: 'wire conductors', 15: 'transmission towers',
16: 'wire structure connectors'}

_singular_classes = {0: '0? seg', 1: '1? seg', 2: 'ground seg', 3: 'low veg seg', 
4: 'med veg seg', 5: 'hi veg seg', 6: 'bldg', 7: 'lo nz seg', 8: 'hi nz seg', 
9: 'wtr seg', 10: 'rail', 11: 'road seg', 12: 'bridge seg', 13: 'w grds seg', 
14: 'w con seg', 15: 'tran twr', 16: 'w str con'}

_horiz_classes = {0: 'uc', 1: 'uc', 2: 'h_patch', 3: 'h_shrub', 4: 'h_bush', 5: 'h_tree', 
6: 'h_roof', 7: 'ln', 8: 'hn', 9: 'h_pond', 10: 'h_rail', 11: 'h_road', 12: 'h_b_deck', 
13: 'h_w_guard', 14: 'h_conductor', 15: 'h_tower_component', 16: 'h_con_component'}

_vert_classes = {0: 'uc', 1: 'uc', 2: 'hill', 3: 'v_patch', 4: 'v_bush', 5: 'v_tree', 
6: 'v_lvl', 7: 'ln', 8: 'hn', 9: 'h_depth', 10: 'v_rail', 11: 'v_road', 12: 'v_b_deck', 
13: 'v_w_guard', 14: 'v_conductor', 15: 'v_tower_component', 16: 'v_con_component'}

#------------------------------------------------------------------------------------------

class AarcCloud():
    def __init__(self, parent:'AarcCloud' = None, las_path: str = None, pt_array:np.ndarray = None,
    cld_name: str = None, tree_lvl: int = None, pt_class = None, lidar_density:float = 0.1,
    below_cld_elevation: float = None, cld_lvl: int = None) -> None:
        if parent is None and las_path is None or parent and las_path:
            raise ValueError('You must provide either a las file: string path, ' +
            'OR a parent cloud: AarcCloud. Only one, not both!')
    
        #-------------child cloud init-----------------------------------------

        if isinstance(parent, AarcCloud) and las_path is None: #sub_cloud goes here
            self.parent = parent
            self.las_path = las_path
            self.pt_array = pt_array
            self.cld_name = cld_name
            self.tree_lvl = tree_lvl
            self.pt_class = pt_class
            self.lidar_density = parent.lidar_density
            self.below_cld_elevation = below_cld_elevation
            self.cld_lvl = cld_lvl


            self.min_el = np.min(self.pt_array[:, 2])
            self.max_el = np.max(self.pt_array[:, 2])

            #need to consider how this is incorporated
            self.concave_hull = None
            self.refined_outline = None
            self.corner_pts = None
            self.polygon = None
            self.mesh = None

        elif las_path is None:
            raise ValueError('You did not provide a valid AarcCloud object.')
        
        #-------------site cloud init------------------------------------------

        elif os.path.isfile(las_path) and las_path.endswith('.las'):
            self.las_file = las_path

            self.lidar_density = lidar_density

            self.cld_name = self._site_name() if cld_name is None else cld_name

            self.tree_lvl = 0 
            
            self._las_pt_array()
            
        
        else: raise ValueError('You did not provide a valid file path or las file.')


    #---------------------getters / setters------------------------------------
    
    @property
    def concave_hull(self):
        return self._concave_hull
    @concave_hull.setter
    def concave_hull(self, new_hull: np.ndarray):
        self._concave_hull = new_hull

    @property
    def refined_outline(self):
        return self._refined_outline
    @refined_outline.setter
    def refined_outline(self, new_ol):
        self._refined_outline = new_ol

    @property
    def corner_pts(self):
        return self._corner_pts
    @corner_pts.setter
    def corner_pts(self, new_crnrs):
        self._corner_pts = new_crnrs

    @property
    def polygon(self):
        return self._polygon
    @polygon.setter
    def polygon(self, new_poly):
        self._polygon = new_poly

    @property
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self, new_mesh):
        self._mesh = new_mesh

    
    #-------------------site cloud methods-------------------------------------
    def _site_name(self):
        file_name = os.path.basename(self.las_file)
        return os.path.splitext(file_name)[0]
    
    def _las_pt_array(self):
        laspy_cld = laspy.read(self.las_file)

        self.pt_class = laspy_cld.classification

        #divide by 100 to maintain meter units
        self.pt_array = np.stack([laspy_cld.X /100, laspy_cld.Y /100,\
             laspy_cld.Z /100], axis= 0).transpose((1,0)) 

        self.min_el = np.min(self.pt_array[:, 2])
        self.max_el = np.max(self.pt_array[:, 2])


    #--------------------general public methods----------------------------------------------------
    def extract_pt_class(self, class_no: int) -> 'AarcCloud':
        """Remove all pts except the specified classification from the point cloud instance.

        Args:
            class_no (int): A valid point classification. See : 
            http://www.asprs.org/a/society/committees/lidar/LAS_1-4_R6.pdf page 10.

        Raises:
            ValueError: Requested point classification was not found in cloud or the cloud has
            already been reduced to a single classification.
        """
        if isinstance(self.pt_class, int):
            raise ValueError(f'This cloud has already been reduced and is solely of class {self.pt_class}.')

        if class_no not in np.unique(self.pt_class):
            raise ValueError(f'Class {class_no} not found in this cloud.' + 
            'Use numpy.unique(<AarcCloud instance>.pt_class) to see available pt classes in this cloud')


        class_mask = (class_no == self.pt_class)
        cloud_name = self._get_cloud_name(class_no)

        return AarcCloud(parent=self, pt_array=self.pt_array[class_mask], cld_name=cloud_name, 
        tree_lvl=self.tree_lvl + 1, pt_class=class_no)

    def cluster_in_plan(self, gap: float, density: int = 20, min_cld_pts: int = 50) -> 'list[AarcCloud]':
        lbls, unique_lbls = self._plan_cluster_labels(self.pt_array, gap, density)

        clustered_clds = []
        dropped_small_clds = 0
        dropped_sparse_clds = 0
        dropped_0_area_clds = 0
        cld_count = 0

        for lbl in unique_lbls:
            cluster_pts = self.pt_array[lbls == lbl]

            if len(cluster_pts) >= min_cld_pts and self._rough_density(cluster_pts) > 0 and \
            self._get_outline_area(cluster_pts) > 0.1:
                cluster_name = self._get_cloud_name(self.pt_class, cld_count, orientation= 'horiz')
                cluster_cld = AarcCloud(parent= self, pt_array=cluster_pts, 
                cld_name=cluster_name, tree_lvl= self.tree_lvl + 1, pt_class=self.pt_class, 
                cld_lvl= self.cld_lvl, below_cld_elevation= self.below_cld_elevation)
                clustered_clds.append(cluster_cld)
                cld_count += 1
            
            if len(cluster_pts) < min_cld_pts: dropped_small_clds += 1
            elif self._rough_density(cluster_pts) == 0: dropped_sparse_clds += 1
            elif self._get_outline_area(cluster_pts) <= 0.1: dropped_0_area_clds += 1

        print(f'You have split {self.cld_name} into {len(unique_lbls)} ' +
            f'{self._get_class_name(self.pt_class)} clusters. {dropped_small_clds} small cloud(s) '+ 
            f'with less than {min_cld_pts} points were dropped. {dropped_sparse_clds} very sparse cloud(s) ' +
            f'were dropped. {dropped_0_area_clds} zero area cloud(s) were dropped.')
        
        return clustered_clds

    def cluster_in_elevation(self, vert_resolution: float = 0.5, smooth: bool = True) -> 'list[AarcCloud]':
        
        #number of points in each elevation bin
        bin_count, elev_bins = self._elevation_count(vert_resolution, smooth)

        #the most frequently occuring elevations - ie. probably major elevations (roofs, etc.)
        common_elev = self._common_elevations(bin_count, elev_bins)

        #flattening pts within vert_resolution to closest freq. occuring elevation
        flat_levels = self._flatten_v_levels(common_elev)

        vertical_clusters = []
        for ix, v_cluster in enumerate(flat_levels):
            cld_nm = self._get_cloud_name(self.pt_class, ix, orientation= 'vert')
            below_el = self.min_el if ix == 0 else np.min(flat_levels[ix - 1][:, 2])
            roof_cld = AarcCloud(parent=self, pt_array=v_cluster, cld_name=cld_nm, 
            tree_lvl= self.tree_lvl + 1, pt_class=self.pt_class, cld_lvl= ix, below_cld_elevation= below_el)

            vertical_clusters.append(roof_cld)
        
        return vertical_clusters


    def remove_vert_srfs(self, vert_tolerance = 0.1) -> None:
        pt_cloud = self._o3d_cloud(self.pt_array)

        og_num = np.shape(self.pt_array)[0]

        pt_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=100))
        normals_z = np.absolute(np.asarray(pt_cloud.normals)[:,2])

        vert_dir_bitmask = np.greater(normals_z, vert_tolerance)
        self.pt_array = self.pt_array[vert_dir_bitmask]

        pt_dif = og_num - np.shape(self.pt_array)[0]
        
        print(f'With a vertical tolerance of {vert_tolerance}, {pt_dif} of {og_num} points were removed.' +
         '\n' + f'This amounts to {pt_dif / og_num * 100} %.')

    def visualize_single_cloud(self) -> None:
        viz_cld = self._o3d_cloud(self.pt_array)
        o3d.visualization.draw_geometries([viz_cld])
    
    def get_o3d_cloud(self) -> 'o3d.geometry.PointCloud':
        o_cloud = self._o3d_cloud(self.pt_array)
        o_cloud.paint_uniform_color([random(), random(), random()])
        return o_cloud


    def calc_concave_hull(self):
        c_hull = ConcaveHull(self.pt_array[:, 0:2])
        self.concave_hull = c_hull.calculate()

    def generate_refined_outline(self):
        if self.concave_hull is  None:
            print(f'{self.cld_name} was unable to generate a concave hull. ' +
            'Attempting to outline all cloud points.')
            try:
                self.refined_outline = RefinedOutline(self.pt_array[:, 0:2], angle_interval= 0.5)
                print(f'Sucessfully generated outline for {self.cld_name}.')

            except:
                raise ValueError(f'Was unable to generate outline for {self.cld_name}')

        else:
            self.refined_outline = RefinedOutline(self.concave_hull, angle_interval= 0.5)

    
    def calc_corner_pts(self):
        try: 
            self.refined_outline.calculate()

            if len(self.refined_outline.intersection_pts) < 3 or self.refined_outline.intersection_pts is None:
                self.refined_outline.simple_calculate()
                print(f'{self.cld_name} was described with a simple outline.')

            else: print(f'{self.cld_name} was described with a detailed outline.')

        except Exception: 
            self.refined_outline.simple_calculate()
            print(f'{self.cld_name} was described with a simple outline.')


        self.corner_pts = self.refined_outline.intersection_pts
        self.polygon = Polygon(self.corner_pts)


    def calc_mesh(self):
        mesh_height = self.below_cld_elevation - self.min_el
        mesh = creation.extrude_polygon(self.polygon, mesh_height)
        mesh.apply_translation([0,0,self.min_el])
        self.mesh = mesh
        

    #------------------------general private methods------------------------------------------------------
    def _plan_cluster_labels(self, cloud, lmnt_gap, density):
        """ Creates label per pt_cloud pt corresponding to a sub-cloud cluster.
            Returns labels and unique labels.
        """

        flat_pts = cloud.copy()
        flat_pts[:, 2] = 0
        flat_cld = self._o3d_cloud(flat_pts)

        #the assigned cluster by point - PARAMETERS ARE DENSITY AND NUM OF POINTS. 
        # SO 1,10 IS A MIN. DENSITY OF 10PTS/SQ.M
        #really in the above, the main parameter that matters is the first one. 
        # It more or less defines the gap between buildings.
        clstr_lbls = np.asarray(flat_cld.cluster_dbscan(lmnt_gap, density))

        return clstr_lbls, np.unique(clstr_lbls)


    def _get_cloud_name(self, class_no, ix = None, orientation = None) -> str:

        if self.tree_lvl == 0:
            return f'{self.cld_name} / {_asprs_classes[class_no]}'

        elif self.tree_lvl == 1 and orientation == 'horiz':
            return f'{self.parent.cld_name} / {_singular_classes[class_no]} {ix}'
        elif self.tree_lvl == 1 and orientation == 'vert':
            raise ValueError('You must cluster in plan at least once before clustering vertically.')

        else:
            if orientation == 'vert':
                return f'{self.cld_name} / {_vert_classes[class_no]} {ix}'
            elif orientation == 'horiz':
                return f'{self.cld_name} / {_horiz_classes[class_no]} {ix}'
            else:
                return f'{self.cld_name} / unoriented clstr {ix}'


    def _get_class_name(self, class_no: int) -> str:
        return _asprs_classes[class_no] if self.tree_lvl == 0 else _singular_classes[class_no]


    @staticmethod
    #converts to o3D cloud, for visualization, normal calc, clustering, etc.
    def _o3d_cloud(pts:np.ndarray) -> 'o3d.geometry.PointCloud':
        pt_cloud = o3d.geometry.PointCloud()
        pt_cloud.points = o3d.utility.Vector3dVector(pts)
        return pt_cloud


    def _elevation_count(self, vert_res: float, smooth: bool) -> 'tuple[np.ndarry]':

        elevs = self.pt_array[:, 2]

        elev_bins = np.arange(self.min_el, self.max_el + vert_res, vert_res)
        bin_indices = (np.digitize(elevs, elev_bins) - 1) #-1 since digitize starts bins at 1 instead of 0

        #how many values in each bin?
        bin_count = np.zeros_like(elev_bins, dtype=int)
        for ix in bin_indices:
            bin_count[ix] = bin_count[ix] + 1 #add 1 every time the index appears in bin_indices

        if smooth: return savgol_filter(bin_count, 5, 3), elev_bins
        else: return bin_count, elev_bins


    def _common_elevations(self, bin_count:np.ndarray, elev_bins:np.ndarray) -> 'tuple[np.ndarray]':
        #assuming that a relevant roof is atleast 5% of the largest roof
        min_roof_count = np.max(bin_count) // 20
        min_roof_cnt_mask = (bin_count >= min_roof_count)
        common_elev_ix = np.arange(len(bin_count))[min_roof_cnt_mask]

        return np.asarray([elev_bins[x] for x in common_elev_ix])
    

    def _flatten_v_levels(self, elevs:np.ndarray) -> 'tuple[np.ndarray]':
        lvl_masks = self._per_lvl_mask(elevs)

        major_lvls = []
        #for each level, we want all points at the level and ABOVE the level
        for ix, mask in enumerate(lvl_masks):
            add_msk = np.zeros_like(mask, dtype= int)
            for msk in lvl_masks[ix:]:
                add_msk = add_msk + msk
            
            cummulative_mask = (add_msk >= 1)

            masked_pts = self.pt_array[cummulative_mask]
            masked_pts[:, 2] = elevs[ix]
            major_lvls.append(masked_pts)

        return major_lvls


    def _per_lvl_mask(self, elevs: np.ndarray) -> np.ndarray:

        # the per level mask - based on pts nearest ot common elevation
        all_elevs = self.pt_array[:, 2]
        lvl_masks = []
        for ix, _ in enumerate(elevs):
            lvl_mask = np.full_like(all_elevs, False)
            for p_ix, pt_elev in enumerate(all_elevs):
                pt_ix = self._find_nearest(elevs, pt_elev)
                if pt_ix == ix:
                    lvl_mask[p_ix] = True
            lvl_masks.append(lvl_mask)

        return lvl_masks


    #finding the closest peak level index for each point
    @staticmethod
    def _find_nearest(array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or fabs(value - array[idx-1]) < fabs(value - array[idx])):
            return idx-1
        else:
            return idx

    @staticmethod
    def _rough_density(pt_list:np.ndarray) -> float:
        """The average points per square unit for a cloud's x/y oriented bounding box

            Only use for VERY rough calculations

        Args:
            pt_list (np.ndarray): 2D cloud points

        Returns:
            float: The average points per square unit in bounding box.
        """
        bounds_x = np.max(pt_list[:,0]) - np.min(pt_list[:,0])
        bounds_y = np.max(pt_list[:,1]) - np.min(pt_list[:,1])
        return len(pt_list[:, 0]) // (bounds_x * bounds_y)

    @staticmethod
    def _get_outline_area(pt_list:np.ndarray) -> float:
        """Get squared unit area of provided 2D outline points. Based on Shoelace formula. See:

        https://stackoverflow.com/a/30408825 and https://en.wikipedia.org/wiki/Shoelace_formula

        Args:
            pt_list (np.ndarray): 2D ordered outline points

        Returns:
            float: The area of the outline in squared units.
        """
        x, y = pt_list[:,0], pt_list[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
