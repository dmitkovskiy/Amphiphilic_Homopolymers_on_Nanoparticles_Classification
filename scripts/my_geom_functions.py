import numpy as np

def rotate_to_principal_axes(points: np.ndarray, max_axes: str = 'Z'):
    """
    Rotates the input points to align their inertia tensor with principal axes.

    Parameters:
    - points (np.ndarray): A 2D array of shape (n, 3) or (n, 4), where n is the number of points. 
                           The first three columns represent the (x, y, z) coordinates. 
                           The optional fourth column represents the type of molecules.
    - max_axes (str): The axis ('Z' or 'X') with maximal inertia moment. Default is Z.
    
    Returns:
    - np.ndarray: A 2D array of rotated points with the same shape as the input array.
    """    
    points_copy = points[:, :3]
    center = points_copy.mean(0)

    normed_points = points_copy - center[None,:]
    gyr = np.einsum('im,in->mn', normed_points,normed_points)/len(points_copy)
    w, v = np.linalg.eig(gyr)

    rot_points = np.einsum('ij,jk',normed_points,v)
    rot_center = np.einsum('ij,jk',np.array(center)[np.newaxis, :],v)
    rot_points = rot_points + rot_center
    
    idx = w.argsort()
    if max_axes == 'Z':
        rot_points = rot_points[:, idx[::-1]]
    elif max_axes == 'X':
        rot_points = rot_points[:, idx]
    else:
        print('The minimum axis is specified incorrectly')

    if len(points[0])>3:
        types_column = points[:, 3]
        rot_points = np.column_stack((rot_points, types_column))

    return rot_points

def drop_points_a_da(points: np.ndarray, a: float, da: float) -> np.ndarray:
    """
    Drops points that are not within the specified distance range from the origin.
    
    Parameters:
    - points (np.ndarray): A 2D array of shape (n, 3) or (n, 4), where n is the number of points. 
                           The first three columns represent the (x, y, z) coordinates. 
                           The optional fourth column represents the type of molecule.
    - a (float): The minimum distance from the origin.
    - da (float): The range of distances to include.
    
    Returns:
    - np.ndarray: A 2D array of points within the specified distance range.
    """     
    distances = np.linalg.norm(points[:, :3], axis=1)
    mask = np.logical_and(distances >= a, distances <= a + da)
    filtered_points = points[mask]
    
    if len(filtered_points) == 0:
        print('There are no points in the specified area')

    return filtered_points

def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """
    Converts Cartesian coordinates to spherical coordinates.
    
    Parameters:
    - points (np.ndarray): A 2D array of shape (n, 3) or (n, 4), where n is the number of points. 
                           The first three columns represent the (x, y, z) coordinates. 
                           The optional fourth column represents the type of molecule.
    
    Returns:
    - np.ndarray: A 2D array of points in spherical coordinates (r, theta, phi).
                  If the input contains a fourth column, it is preserved in the output.
    """     
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(np.sum(points[:, :3]**2, axis=1))
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    spherical_points = np.column_stack((r, theta, phi))

    if len(points[0])>3:
        spherical_points = np.column_stack((spherical_points, points[:, 3] ))
    
    return spherical_points

def spherical_to_cartesian(points):
    """
    Converts spherical coordinates to Cartesian coordinates.
    
    Parameters:
    - points (np.ndarray): A 2D array of shape (n, 3) or (n, 4), where n is the number of points. 
                           The first three columns represent the (r, theta, phi) coordinates. 
                           The optional fourth column represents the type of molecule.
    
    Returns:
    - np.ndarray: A 2D array of points in Cartesian coordinates (x, y, z).
                  If the input contains a fourth column, it is preserved in the output.
    """
    r, theta, phi = points[:, 0], points[:, 1], points[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    cartesian_points = np.column_stack((x, y, z))

    if len(points[0])>3:
        cartesian_points = np.column_stack((cartesian_points, points[:, 3]))

    return cartesian_points

def flatten_points(points, a):
    """
    Flattens the points onto a sphere by setting the radial distance to a constant value.
    
    Parameters:
    - points (np.ndarray): A 2D array of shape (n, 3) or (n, 4), where n is the number of points. 
                           The first three columns represent the (x, y, z) coordinates. 
                           The optional fourth column represents the type of molecule.
    - a (float): The constant radial distance to set.
    
    Returns:
    - np.ndarray: A 2D array of flattened points in Cartesian coordinates.
                  If the input contains a fourth column, it is preserved in the output.
    """     
    spherical_points = cartesian_to_spherical(points)
    spherical_points[:, 0] = a
    cartesian_coords = spherical_to_cartesian(spherical_points)

    return cartesian_coords

def rotate_drop_flatten(points, a= 4, da= 6):
    """
    Rotates points to principal axes, drops points not within the specified distance range, and flattens the remaining points.
    
    Parameters:
    - points (np.ndarray): A 2D array of shape (n, 3) or (n, 4), where n is the number of points. 
                           The first three columns represent the (x, y, z) coordinates. 
                           The optional fourth column represents the type of molecule.
    - a (float): The minimum distance from the origin. Default is 4.
    - da (float): The range of distances to include. Default is 6.
    
    Returns:
    - np.ndarray: A 2D array of transformed points, or an empty array if no points are within the specified distance range.
    """    
    r_points = rotate_to_principal_axes(points)
    d_r_points = drop_points_a_da(r_points, a, da)
    if len(d_r_points)>0:
        f_d_r_points = flatten_points(d_r_points, a)
        return f_d_r_points
    else:
        return []
