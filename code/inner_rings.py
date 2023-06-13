from math import floor, ceil
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree as KTREE

class InnerRings:
    def __init__(self, pt_array:np.ndarray, voxel_size: int = 5) -> None:
        self.pt_array = pt_array
        self.min_vals = np.min(pt_array, axis= 0)
        self.max_vals = np.max(pt_array, axis= 0)

        self.voxel_size = voxel_size

        self.x_range, self.y_range = _voxel_pad(self.min_vals, self.max_vals, voxel_size)

        self.lowest_pt = _start_pt(pt_array)


    def calculate(self):
        bit_mask, x_edges, y_edges = _bitmask(self.pt_array, self.x_range, self.y_range)
        void_points = _void_array(x_edges, y_edges, bit_mask)

        clustered_pts = DBSCAN(eps= self.voxel_size, min_samples= 3).fit(void_points)
        cluster_labels = clustered_pts.labels_

    

        return [self.lowest_pt] if len(np.unique(cluster_labels)) <= 2 else  \
        _pt_cluster_list(cluster_labels, void_points, self.lowest_pt, self.pt_array)

def _start_pt(pts: np.ndarray, highest_pt = False) -> np.ndarray:
    pt_ix = -1 if highest_pt else 0
    index = np.argsort(pts[:, 1])[pt_ix]
    return pts[index]

def _voxel_pad(min_val: np.ndarray, max_val: np.ndarray, vx_size: float) -> 'tuple[np.ndarray]':
    #round range to multiple of voxel size and add padding
    vox_max = np.ceil(max_val / vx_size) * vx_size + vx_size * 2
    vox_min = np.floor(min_val / vx_size) * vx_size - vx_size * 2
    return np.arange(vox_min[0], vox_max[0] + 1, vx_size), \
           np.arange(vox_min[1], vox_max[1] + 1, vx_size)

def _bitmask(pts:np.ndarray, x_range:np.ndarray, y_range:np.ndarray) -> np.ndarray:
        #creating the range of values
        pt_voxels, y_edges, x_edges = np.histogram2d(pts[:, 0], pts[:, 1], bins=(x_range, y_range))

        #creating the test zeros array
        zero_array = np.zeros(pt_voxels.shape)
        #creating the bit mask
        return np.greater(pt_voxels, zero_array), x_edges, y_edges

def _void_pts(bins: np.ndarray) -> 'list[float]':
    return [float(np.mean([bins[i], bins[i + 1]])) for i in range(len(bins) - 1)]

def _void_array(x_edge:np.ndarray, y_edge:np.ndarray, bit_mask:np.ndarray) -> 'tuple[np.ndarray]':
    void_x, void_y = np.meshgrid(_void_pts(x_edge), _void_pts(y_edge), indexing='xy')

    void_x[bit_mask] = np.nan
    void_y[bit_mask] = np.nan

    #DBSCAN does not accept NAN values, so we need to drop them from our x_void / y_void arrays first.

    void_x = void_x.flatten()
    void_y = void_y.flatten()

    void_x = void_x[~np.isnan(void_x)]
    void_y = void_y[~np.isnan(void_y)]

    void_x = void_x.reshape(-1,1)
    void_y = void_y.reshape(-1,1)

    return np.concatenate((void_y, void_x), axis=1)

def _closest_pt(pt_cloud: np.ndarray, search_pt:np.ndarray) -> np.ndarray:
    search_tree = KTREE(pt_cloud)
    pt_dist, pt_index = search_tree.query(search_pt)
    return pt_cloud[pt_index, :]

def _pt_cluster_list(labels:np.ndarray, void_points:np.ndarray, low_pt: np.ndarray, pt_cloud: np.ndarray) -> list:
    start_pts = [low_pt]

    for label in np.unique(labels):
        if label > 0:
            bit_mask = (labels == label)
            cluster = void_points[bit_mask]
            void_start = _start_pt(cluster, highest_pt= True)
            start_pts.append(_closest_pt(pt_cloud, void_start))

    return start_pts
        