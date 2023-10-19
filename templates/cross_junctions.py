import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from matplotlib.path import Path

# You may add support functions here, if desired.

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    
    # Construct the A matrix
    A = []
    for i in range(4):
        x, y = I1pts[0][i], I1pts[1][i]
        u, v = I2pts[0][i], I2pts[1][i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)

    # Solve for h using the nullspace of A
    h = null_space(A)

    # Reshape h into 3x3 H homography matrix
    H = h.reshape(3, 3)

    # Normalize H by scaling all entries such that the lower right entry  is 1
    H /= H[-1, -1]

    #------------------

    return H, A

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
 
    # The linear least squares problem is defined as minimizing ||Ax-b||^2
    # where A is mxn matrix with m the number of data points and n is the number of variables
    # x is the vector of variables we are solving for
    # b is a vector of length m, representing the observed data points
    
    # Set up the A matrix
    h, l = I.shape
    A = []

    for y in range(h):
        for x in range(l):
            A.append([x**2, x*y, y**2, x, y, 1])

    A = np.array(A)

    # Set up the b matrix
    b = I.flatten()

    # Solve the linear least squares problem
    alpha, beta, gamma, delta, epsilon, zeta = lstsq(A, b, rcond=None)[0]

    # Get the coordinates of the saddle point by finding the intersection of 2 lines
    pt = -inv(np.array([[2*alpha, beta], [beta, 2*gamma]])) @ np.array([[delta], [epsilon]])

    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---

    n = Wpts.shape[1]
    Ipts = []
    square_dim = Wpts[1][0] - Wpts[0][0]

    # Estimate the location of bounding box corners in world frame
    top_left = np.array([[min(Wpts[0]) - (1.5*square_dim)], [min(Wpts[1] - (1.15*square_dim))]])
    top_right = np.array([[max(Wpts[0]) - (1.25*square_dim)], [min(Wpts[1] - (1.15*square_dim))]])
    bottom_right = np.array([[max(Wpts[0]) - (1.25*square_dim)], [max(Wpts[1] - (1.15*square_dim))]])
    bottom_left = np.array([[min(Wpts[0]) - (1.5*square_dim)], [max(Wpts[1] - (1.15*square_dim))]])
    bworld = np.hstack((top_left, top_right, bottom_right, bottom_left))

    # Compute homography matrix
    H, A = dlt_homography(bworld, bpoly)

    window = 10
    for i in range(n):
        # Use homography matrix to find corresponding points
        point = np.array([[Wpts[0][i], Wpts[1][i], 1]]).T
        corresponding_point = H @ point
        corresponding_point /= corresponding_point[-1]

        # Set up patch around point
        xmin = int(np.round(corresponding_point[0]-window))
        xmax = int(np.round(corresponding_point[0]+window))
        ymin = int(np.round(corresponding_point[1]-window))
        ymax = int(np.round(corresponding_point[0]+window))
        patch = I[ymin:ymax+1, xmin:xmax+1]

        # Find saddle point in the patch
        saddlept = saddle_point(patch)
        saddlept += np.array([xmin, ymin]).T

        Ipts.append(saddlept)

    Ipts = np.array(Ipts).T

    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts