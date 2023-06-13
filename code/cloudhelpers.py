import numpy as np

def adjust_building_angles(bldg_clds: list, angle_tolerance: float = 3.0):

    for cld in bldg_clds:
        if cld.refined_outline is None:
            raise ValueError('You must generate refined outlines before adjusting angles')

    pts_cnt_per_cld = [len(x.pt_array) for x in bldg_clds]
    dom_cld = [cld for _, cld in sorted(zip(pts_cnt_per_cld, bldg_clds))][-1]
    guide_angles = dom_cld.refined_outline.dom_angles

    for cld in bldg_clds:
        for ix, angle in enumerate(cld.refined_outline.dom_angles):
            close_guide_angle = find_nearest(guide_angles, angle)
            if abs(close_guide_angle - angle) <= angle_tolerance:
                print(f'adjusted {cld.cld_name} angle {angle} to:')
                cld.refined_outline.dom_angles[ix] = close_guide_angle
                print(cld.refined_outline.dom_angles[ix])

def find_nearest(array:np.ndarray, value):
    closest_ix = (np.abs(array - value)).argmin()
    return array[closest_ix]

#------------------------------------------------------------------------------------------------------

def adjust_building_corners(bldg_clds: list, crnr_dist_tolerance: float = 1.0):
    for cld in bldg_clds:
        if cld.corner_pts is None:
            raise ValueError('You need to generate corner points for each cloud before proceeding.')
    
    for ix, cld in enumerate(bldg_clds):
        if ix != 0:
            for px, pt in enumerate(cld.corner_pts):
                distances = pt_2_pt_distance(pt, prev_pts)
                if np.min(distances) <= crnr_dist_tolerance:
                    cld.corner_pts[px] = prev_pts[np.argmin(distances)]
                    print(f'adjusted {cld.cld_name} corner point {px}')

        prev_pts = cld.corner_pts


def pt_2_pt_distance(pt_a:np.ndarray, pt_cld:np.ndarray):
    """Return distance(s) from 1 pt to one or multiple points

    Args:
        pt_a (np.ndarray): 2D point to measure from
        pt_cld (np.ndarray): 2D point(s) to measure to

    Returns:
        _type_: Distance(s) from pt_a to the pt(s) in the cloud
    """
    return np.sqrt(np.square(pt_cld[:,0] - pt_a[0])+ np.square(pt_cld[:,1] - pt_a[1]))

#-----------------------------------------------------------------------------------------------------

def adjust_foot_prints(bldg_clds: list):
    for cld in bldg_clds:
        if cld.polygon is None:
            raise ValueError('you need to generate corner points for each cloud before proceeding.')
    
    for cx, cld in enumerate(bldg_clds):
        if cx != 0:
            intersect_poly = cld.polygon.intersection(prev_poly)
            if intersect_poly.exterior.area != 0:
                cld.polygon = intersect_poly

        prev_poly = cld.polygon
