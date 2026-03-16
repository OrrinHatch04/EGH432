from typing import Literal as L, List
import numpy as np

# --------- Question 1.1 ---------- #


def is_SO2(rot: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Checks if the input matrix is a SO(2) matrix.

    This method checks if the input array is a valid SO(2) matrix,
    to the numerical threshold of `tol`. 'tol' is the numerical error
    tolerance for each key characteristic of a matrix in the Special
    Orthogonal Group and is calculated using the vector norm.
    hint: np.linalg.norm(X) < tol checks if X is less than tol. 

    Parameters
    ----------
    rot
        the input array to check - size is undefined
    tol
        the error tolerance for the numerical checks

    Returns
    -------
    is_SO2
        True if the input array is a valid SO(2) matrix, False otherwise
    """

    if rot.shape != (2,2):
        return False
    
    orthogonality = np.linalg.norm(rot.T @ rot - np.eye(2)) < tol
    determinant = np.linalg.norm(np.linalg.det(rot) - 1) < tol
    
    return bool(orthogonality and determinant) 


# --------- Question 1.2 ---------- #


def is_SO3(rot: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Checks if the input matrix is a SO(3) matrix.

    This method checks if the input array is a valid SO(3) matrix,
    to the numerical threshold of `tol`. 'tol' is the numerical error
    tolerance for each key characteristic of a matrix in the Special
    Orthogonal Group and is calculated using the vector norm.


    Parameters
    ----------
    rot
        the input array to check - size is undefined
    tol
        the error tolerance for the numerical checks

    Returns
    -------
    is_SO3
        True if the input array is a valid SO(3) matrix, False otherwise
    """

    if rot.shape != (3,3):
        return False
    
    orthogonality = np.linalg.norm(rot.T @ rot - np.eye(3)) < tol
    determinant = np.linalg.norm(np.linalg.det(rot) - 1) < tol
    
    return bool(orthogonality and determinant)


# --------- Question 1.3 ---------- #


def RPY_to_SO3(rpy: np.ndarray) -> np.ndarray:
    """
    Converts intrinsic ZYX roll-pitch-yaw angles to an SO(3) matrix.

    This method converts a set of RPY angles with ZYX ordering to
    an SO(3) matrix.

    Rotate by yaw about the z-axis, then by pitch about the
    new y-axis, then by roll about the newest x-axis.

    Parameters
    ----------
    rpy
        a (3, 1) array of with the order [roll, pitch, yaw]

    Returns
    -------
    rot
        the corresponding SO(3) matrix

    """

    roll  = rpy[0, 0]
    pitch = rpy[1, 0]
    yaw   = rpy[2, 0]
    
    Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ], dtype=float)
    Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ], dtype=float)
    Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=float)
    
    rot = Rz @ Ry @ Rx
    
    return rot
    


# --------- Question 2.1 ---------- #


def is_eulvec(eulvec: np.ndarray) -> bool:
    """
    Checks if the input array is a valid Euler vector.

    This method checks if the input array is a valid Euler vector, where
    the Euler vector must be a (3, 1) numpy array.

    Parameters
    ----------
    eulvec
        the input array to check - size is undefined

    Returns
    -------
    is_eulvec
        True if the input array is a valid Euler vector, False otherwise
    """

    return bool(isinstance(eulvec, np.ndarray) and eulvec.shape == (3, 1))


# --------- Question 2.2 ---------- #


def is_angax(angax: np.ndarray, tol=1e-6) -> bool:
    """
    Checks if the input array is a valid angle-axis vector.

    This method checks if the input array is a valid angle-axis vector,
    where the angle-axis vector must be a (4, 1) numpy array:

    [
        [theta],
        [etahat_1],
        [etahat_2],
        [etahat_3]
    ]

    where theta is the rotation angle and etahat is the rotation axis. The
    error threshold for the vector norm of the rotation axis is given by
    `tol`.

    Parameters
    ----------
    angax
        the input array to check - size is undefined
    tol
        the error threshold for the norm of the rotation axis

    Returns
    -------
    is_angax
        True if the input array is a valid angle-axis vector, False otherwise
    """

    if angax.shape != (4, 1):
        return False
    
    axis = angax[1:]
    
    return bool(np.linalg.norm(np.linalg.norm(axis) - 1) < tol)


# --------- Question 2.3 ---------- #


def angax_to_eulvec(angax: np.ndarray) -> np.ndarray:
    """
    Converts an angle-axis vector to an Euler vector.

    This method converts an angle-axis vector to an Euler vector.

    Parameters
    ----------
    angax
        the input angle-axis vector as a (4, 1) numpy array as described by
        the `is_angax` method

    Returns
    -------
    eulvec
        the corresponding Euler vector as a (3, 1) numpy array
    """

    theta = angax[0, 0]
    axis = angax[1:]
    
    return theta * axis


# --------- Question 2.4 ---------- #


def eulvec_to_SO3(eulvec: np.ndarray) -> np.ndarray:
    """
    Converts a Euler vector to a SO(3) matrix.

    This method converts a Euler vector to a SO(3) matrix.

    Parameters
    ----------
    eulvec
        the input Euler vector as a (3, 1) numpy array as described by the
        `is_eulvec` method

    Returns
    -------
    rot
        the corresponding SO(3) matrix as a (3, 3) numpy array
    """

    theta = np.linalg.norm(eulvec)
    
    if theta < 1e-10:
        return np.eye(3)
    
    kx, ky, kz = (eulvec / theta).flatten()
    c, s = np.cos(theta), np.sin(theta)

    # Combine Skew-Symmetric Matrix with Rodriquez Formula intrinsicly to increase computation speed.
    return np.array([
        [c + kx**2*(1-c),      kx*ky*(1-c) - kz*s,  kx*kz*(1-c) + ky*s],
        [kx*ky*(1-c) + kz*s,   c + ky**2*(1-c),      ky*kz*(1-c) - kx*s],
        [kx*kz*(1-c) - ky*s,   ky*kz*(1-c) + kx*s,   c + kz**2*(1-c)  ]
    ])


# --------- Question 3.1 ---------- #


def is_quat(quat: np.ndarray, tol=1e-6) -> bool:
    """
    Checks if the input array is a valid unit quaternion.

    This method checks if the input array is a valid unit quaternion,
    where the unit quaternion must be a (4, 1) numpy array:

    [
        [s],
        [v_x],
        [v_y],
        [v_z]
    ]

    where s is the real component and v_x, v_y, and v_z are the imaginary
    components. The error threshold for the vector norm of the quaternion is
    given by `tol`.

    Parameters
    ----------
    quat
        the input array to check as a (4, 1) numpy array
    tol
        the error threshold for the norm of the quaternion

    Returns
    -------
    is_quat
        True if the input array is a valid quaternion, False otherwise
    """

    if quat.shape != (4, 1):
        return False

    return bool(np.linalg.norm(np.linalg.norm(quat) - 1) < tol)


# --------- Question 3.2 ---------- #


def quat_to_SO3(quat: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion to a SO(3) matrix.

    This method converts a quaternion to a SO(3) matrix where the quaternion is
    given as described by `is_quat`.

    Parameters
    ----------
    quat
        the input quaternion as a (4, 1) numpy array

    Returns
    -------
    rot
        the corresponding SO(3) matrix as a (3, 3) numpy array
    """

    s = quat[0, 0]
    vx, vy, vz = quat[1, 0], quat[2, 0], quat[3, 0]

    # Precompute repeated terms to save computation
    vx2, vy2, vz2 = vx**2, vy**2, vz**2
    vxy, vxz, vyz = vx*vy, vx*vz, vy*vz
    svx, svy, svz = s*vx, s*vy, s*vz

    return np.array([
        [1 - 2*(vy2 + vz2),  2*(vxy - svz),    2*(vxz + svy)],
        [2*(vxy + svz),       1 - 2*(vx2+vz2),  2*(vyz - svx)],
        [2*(vxz - svy),       2*(vyz + svx),    1 - 2*(vx2+vy2)]
    ])


# --------- Question 3.3 ---------- #


def SO3_to_quat(rot: np.ndarray) -> np.ndarray:
    """
    Converts a SO(3) matrix to a quaternion.

    This method converts a SO(3) matrix to a quaternion.

    Parameters
    ----------
    rot
        the input SO(3) matrix as a (3, 3) numpy array

    Returns
    -------
    quat
        the corresponding quaternion as a (4, 1) numpy array as described by
        the `is_quat` method
    """

    trace = rot[0,0] + rot[1,1] + rot[2,2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1)
        w = 0.25 / s
        x = (rot[2,1] - rot[1,2]) * s
        y = (rot[0,2] - rot[2,0]) * s
        z = (rot[1,0] - rot[0,1]) * s
    elif rot[0,0] > rot[1,1] and rot[0,0] > rot[2,2]:
        s = 2 * np.sqrt(1 + rot[0,0] - rot[1,1] - rot[2,2])
        w = (rot[2,1] - rot[1,2]) / s
        x = 0.25 * s
        y = (rot[0,1] + rot[1,0]) / s
        z = (rot[0,2] + rot[2,0]) / s
    elif rot[1,1] > rot[2,2]:
        s = 2 * np.sqrt(1 + rot[1,1] - rot[0,0] - rot[2,2])
        w = (rot[0,2] - rot[2,0]) / s
        x = (rot[0,1] + rot[1,0]) / s
        y = 0.25 * s
        z = (rot[1,2] + rot[2,1]) / s
    else:
        s = 2 * np.sqrt(1 + rot[2,2] - rot[0,0] - rot[1,1])
        w = (rot[1,0] - rot[0,1]) / s
        x = (rot[0,2] + rot[2,0]) / s
        y = (rot[1,2] + rot[2,1]) / s
        z = 0.25 * s

    return np.array([[w], [x], [y], [z]])


# --------- Question 4.1 ---------- #


def orientation_type(arg: np.ndarray, tol=1e-6) -> L["quat", "eulvec", "SO3"]:
    """
    Tests if the input is a valid 3D orientation representation

    This method tests if the input array `arg` is either a valid
    quaternion, Euler vector, or SO(3) matrix.

    Parameters
    ----------
    arg
        The input array to test

    Returns
    -------
    type
        A string indicating the type of orientation representation, either
        'quat', 'eulvec', or 'SO3'

    Raises
    ------
    ValueError
        if the input array is not a valid quaternion, Euler vector,
        or SO(3) matrix

    """

    if is_SO3(arg, tol):
        return "SO3"
    elif is_quat(arg, tol):
        return "quat"
    elif is_eulvec(arg):
        return "eulvec"
    else:
        raise ValueError(f"Input is not a valid orientation representation")


# --------- Question 4.2 ---------- #


def filter_orientations(rots: List[np.ndarray], tol=1e-6) -> List[np.ndarray]:
    """
    Filters a list of orientations and removes errors

    The method takes in orientations where the type may be
    a quaternion, Euler vector, SO(3) matrix or an error. It must return
    a new list of valid orientations as a consistent type SO(3) matrix

    Parameters
    ----------
    rots
        A list of orientations where each orientation may be a quaternion,
        Euler vector, SO(3) matrix, or an error

    Returns
    -------
    rots
        A list of numpy arrays where each array is a valid orientation as a
        consistent SO(3) type (errors should be removed)
    """

    result = []
    for rot in rots:
        try:
            otype = orientation_type(rot, tol)
            if otype == "SO3":
                result.append(rot)
            elif otype == "quat":
                result.append(quat_to_SO3(rot))
            elif otype == "eulvec":
                result.append(eulvec_to_SO3(rot))
        except ValueError:
            pass  # skip invalid entries
    return result


# --------- Question 4.3 ---------- #


def get_planet_orienations(rots: List[np.ndarray]) -> List[np.ndarray]:
    """
    Gets planet orientations realtive to Earth

    Given a list of converted planet orientations, as output from the
    `filter_orientations` method, this method returns the planet
    orientations relative to Earth (as the same orientation type as output by
    `filter_orientations`).

    Parameters
    ----------
    rots
        A list of relative planet orientations as output by `filter_orientations`
        Each element of the list is orientation of a planet relative to the previous planet

    Returns
    -------
    planet_rots
        A list of planet orientations relative to Earth where the orientation type
        is the same as the input `rots`
    """

    result = []
    accumulated = np.eye(3)
    for rot in rots:
        accumulated = accumulated @ rot
        result.append(accumulated)
    return result
