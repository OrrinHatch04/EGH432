from typing import Tuple
import numpy as np
import spatialmath as sm
import spatialmath.base as smb
import roboticstoolbox as rtb
import sympy as sym

# --------- Question 1.1 ---------- #

def is_twist(twist: np.ndarray) -> bool:
    return isinstance(twist, np.ndarray) and twist.shape == (6,)


# --------- Question 1.2 ---------- #

def SE3_to_twist(T: np.ndarray) -> np.ndarray:
    return smb.trlog(T, twist=True)


# --------- Question 1.3 ---------- #

def twist_to_SE3(twist: np.ndarray) -> np.ndarray:
    return smb.trexp(twist)


# --------- Question 2.1 ---------- #

def SE3_twist_traj(wTa: sm.SE3, wTb: sm.SE3, n: int) -> sm.SE3:
    xi_a = SE3_to_twist(wTa.A)
    xi_b = SE3_to_twist(wTb.A)
    traj_data = rtb.mtraj(rtb.quintic, xi_a, xi_b, n)

    traj = sm.SE3.Empty()
    for q in traj_data.q:
        traj.append(sm.SE3(twist_to_SE3(q)))
    return traj

# --------- Question 2.2 ---------- #

def SE3_traj_relative(traj_world: sm.SE3) -> sm.SE3:
    traj_relative = sm.SE3.Empty()
    for i in range(1, len(traj_world)):
        rel = traj_world[i - 1].inv() * traj_world[i]
        traj_relative.append(rel)
    return traj_relative


# --------- Question 3.1 ---------- #

def quintic_properties_symbolic(
    coeffs: Tuple[float, float, float, float, float, float]
) -> Tuple[float, float]:

    A, B, C, D, E, F = coeffs
    acc_c  = [20*A, 12*B,  6*C, 2*D]
    jerk_c = [60*A, 24*B,  6*C]
    snap_c = [120*A, 24*B]

    def max_abs(poly, deriv):
        roots = np.roots(deriv)
        pts = [0.0, 1.0] + [r.real for r in roots
                             if abs(r.imag) < 1e-10 and 0.0 <= r.real <= 1.0]
        return float(max(abs(np.polyval(poly, t)) for t in pts))

    return max_abs(acc_c, jerk_c), max_abs(jerk_c, snap_c)


# --------- Question 4.1 ---------- #

def rocket_diagnostics(traj: sm.SE3, T: sm.SE3) -> float:
    target = T.A
    current = np.eye(4)

    for i, pose in enumerate(traj):
        current = current @ pose.A
        if np.allclose(current, target, atol=1e-4):
            return i * 0.5

    return None

# --------- Question 4.2 ---------- #

def rocket_pose(traj: sm.SE3, wTa: sm.SE3) -> Tuple[sm.SE3, float]:
    target = wTa.A
    wTg = traj[-1]

    a_idx = next(
        i for i in range(len(traj))
        if np.allclose(traj[i].A, target, atol=1e-8)
    )

    return wTa.inv() * wTg, (len(traj) - 1 - a_idx) * 0.5

