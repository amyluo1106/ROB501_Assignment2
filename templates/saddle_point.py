import numpy as np
from numpy.linalg import inv, lstsq

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

    for x in range(l):
        for y in range(h):
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