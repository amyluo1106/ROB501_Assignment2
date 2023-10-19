import numpy as np
from numpy.linalg import inv

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """

    #--- FILL ME IN ---
    # Extract rotation matrix from homogenous pose matrix
    R = Twc[:3, :3]
    # Extract translation vector from homogenous pose matrix
    t = Twc[:3, -1:]

    # Using pinhole projection model to find image plane coordinates of the projection of 3D landmark world point
    Ipt = K @ R.T @ (Wpt-t)

    # Find partial derivatives of x, y, z
    dxyz = K @ (R.T) @ (-np.eye(3))

    # Convert rotation matrix to euler angles
    eulerAngles = rpy_from_dcm(R) 

    # phi
    roll = eulerAngles[0]
    # theta
    pitch = eulerAngles[1]
    # psi
    yaw = eulerAngles[2]

    # Set up rotation matrices for euler angles
    Cyaw = dcm_from_rpy([0, 0, yaw])
    Cpitch = dcm_from_rpy([0, pitch, 0])
    Croll = dcm_from_rpy([roll, 0, 0])

    # Find partial derivatives of yaw, pitch, roll
    y1 = inv(R) @ (Wpt-t)
    dyaw = -inv(R) @ [[0, -1, 0], [1, 0, 0], [0, 0, 0]] @ Cyaw @ Cpitch @ Croll @ y1
    dpitch = -inv(R) @ Cyaw @ [[0, 0, 1], [0, 0, 0], [-1, 0, 0]] @ Cpitch @ Croll @ y1
    droll = -inv(R) @ Cyaw @ Cpitch @ [[0, 0, 0], [0, 0, -1], [0, 1, 0]] @ Croll @ y1

    # Assemble Jacobian
    J = np.hstack((dxyz, K @ droll, K @ dpitch, K @ dyaw))

    # Divide through by z
    z = Ipt[-1, 0]
    J = (z * J - Ipt @ J[-1:, :]) / (z*z)
    J = J[0:2]

    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J

def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])

import numpy as np

def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy