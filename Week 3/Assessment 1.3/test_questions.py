"""
Pytest test suite for questions.py — EGH432 Week 3 Assessment 1.3

Analytical references:
  - Corke, P. (2023). Robotics, Vision and Control (3rd ed.). Springer.
  - Lynch & Park (2017). Modern Robotics. Cambridge University Press.
  - QUT Robotics Academy course material.

Test philosophy:
  - Every function is tested for correctness (hand-calculated / closed-form answers),
    for robustness (edge cases), and for runtime (inline timing asserts).
  - Fixtures are module-scoped where possible so expensive SE3_twist_traj calls
    are paid only once per test session — keeping the full suite fast.
"""

import time
from typing import Tuple

import numpy as np
import pytest
import spatialmath as sm
import spatialmath.base as smb

from questions import (
    SE3_traj_relative,
    SE3_to_twist,
    SE3_twist_traj,
    is_twist,
    quintic_properties_symbolic,
    rocket_diagnostics,
    rocket_pose,
    twist_to_SE3,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def pytest_collection_modifyitems(items):
    """Print question section headers during collection."""
    section_map = {
        "TestIsTwist":                  "Q1.1 — is_twist",
        "TestSE3ToTwist":               "Q1.2 — SE3_to_twist",
        "TestTwistToSE3":               "Q1.3 — twist_to_SE3",
        "TestSE3TwistTraj":             "Q2.1 — SE3_twist_traj",
        "TestSE3TrajRelative":          "Q2.2 — SE3_traj_relative",
        "TestQuinticPropertiesSymbolic":"Q3.1 — quintic_properties_symbolic",
        "TestRocketDiagnostics":        "Q4.1 — rocket_diagnostics",
        "TestRocketPose":               "Q4.2 — rocket_pose",
    }
    for item in items:
        cls = item.cls.__name__ if item.cls else None
        if cls in section_map:
            item._nodeid = item._nodeid.replace(
                cls, f"{section_map[cls]}/{cls}"
            )

def _make_se3(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0) -> sm.SE3:
    """Build an SE3 from translation + intrinsic ZYX Euler angles (radians)."""
    return sm.SE3.Trans(tx, ty, tz) * sm.SE3.Rz(rz) * sm.SE3.Ry(ry) * sm.SE3.Rx(rx)


def _assert_valid_se3(T: np.ndarray, atol: float = 1e-8) -> None:
    """Assert that a 4×4 numpy array is a valid SE3 matrix."""
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=atol,
                               err_msg="Rotation block is not orthogonal")
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=atol,
                               err_msg="Rotation block det != 1")
    np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=atol,
                               err_msg="Bottom row is not [0,0,0,1]")


# Module-scoped trajectories (computed once for the whole session)
@pytest.fixture(scope="module")
def world_traj_21() -> sm.SE3:
    """21-pose world-frame trajectory wTa→wTb (step dt=0.5 s → total 10 s)."""
    wTa = sm.SE3()
    wTb = _make_se3(6.0, 3.0, 1.0, 0.0, 0.0, np.pi / 3)
    return SE3_twist_traj(wTa, wTb, 21)


@pytest.fixture(scope="module")
def world_traj_11() -> sm.SE3:
    """11-pose world-frame trajectory wTa→wTb (step dt=0.5 s → total 5 s)."""
    wTa = sm.SE3()
    wTb = _make_se3(4.0, 0.0, 2.0, 0.0, 0.0, np.pi / 4)
    return SE3_twist_traj(wTa, wTb, 11)


@pytest.fixture(scope="module")
def rel_traj_10(world_traj_11) -> sm.SE3:
    """Prepended relative traj: [world[0], rel[0], ..., rel[9]] (11 elements total)."""
    traj = sm.SE3.Empty()
    traj.append(world_traj_11[0])
    for r in SE3_traj_relative(world_traj_11):
        traj.append(r)
    return traj


# ═════════════════════════════════════════════════════════════════════════════
# Q1.1 — is_twist
# ═════════════════════════════════════════════════════════════════════════════

class TestIsTwist:
    """is_twist must return True iff the input is a numpy array with shape (6,)."""

    # --- Positive cases ---

    def test_ones(self):
        assert is_twist(np.ones(6)) is True

    def test_zeros(self):
        assert is_twist(np.zeros(6)) is True

    def test_float64(self):
        assert is_twist(np.array([1.0, -2.5, 0.0, 0.1, 0.0, 3.14])) is True

    def test_int_dtype(self):
        # dtype should not matter — only shape
        assert is_twist(np.array([1, 2, 3, 4, 5, 6])) is True
    
    def test_float32_dtype(self):
        # dtype should not matter — only shape and type
        assert is_twist(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)) is True

    def test_large_values(self):
        assert is_twist(np.array([1e9, -1e9, 0, 1e-9, 0, 0])) is True

    @pytest.mark.parametrize("v", [
        np.array([0, 0, 0, 0, 0, np.pi]),       # pure rotation z
        np.array([1, 0, 0, 0, 0, 0], dtype=float),  # pure translation x
        np.random.default_rng(0).random(6),
    ])
    def test_parametrized_valid(self, v):
        assert is_twist(v) is True

    # --- Negative cases: wrong shape ---

    def test_shape_6_1(self):
        assert is_twist(np.zeros((6, 1))) is False

    def test_shape_1_6(self):
        assert is_twist(np.zeros((1, 6))) is False

    def test_shape_5(self):
        assert is_twist(np.zeros(5)) is False

    def test_shape_7(self):
        assert is_twist(np.zeros(7)) is False

    def test_shape_3(self):
        assert is_twist(np.zeros(3)) is False

    def test_shape_empty(self):
        assert is_twist(np.zeros(0)) is False

    def test_2d_3x2(self):
        assert is_twist(np.zeros((3, 2))) is False

    # --- Negative cases: wrong type ---

    def test_list(self):
        assert is_twist([1, 2, 3, 4, 5, 6]) is False

    def test_tuple(self):
        assert is_twist((1, 2, 3, 4, 5, 6)) is False

    def test_none(self):
        assert is_twist(None) is False

    def test_int(self):
        assert is_twist(6) is False

    def test_float_scalar(self):
        assert is_twist(1.0) is False

    def test_string(self):
        assert is_twist("123456") is False

    # --- Speed ---

    def test_speed_10k_calls(self):
        """10 000 calls must complete in under 50 ms."""
        v = np.ones(6)
        t0 = time.perf_counter()
        for _ in range(10_000):
            is_twist(v)
        assert time.perf_counter() - t0 < 0.05


# ═════════════════════════════════════════════════════════════════════════════
# Q1.2 — SE3_to_twist
# ═════════════════════════════════════════════════════════════════════════════

class TestSE3ToTwist:
    """
    Reference: Corke §2.3, Lynch & Park §3.3 — matrix logarithm of SE3.

    For a pure translation T(t):   twist = [tx, ty, tz, 0, 0, 0]
    For a pure rotation Rot(ω, θ): twist = [0, 0, 0, ωx·θ, ωy·θ, ωz·θ]
    """

    def test_identity_is_zero_twist(self):
        twist = SE3_to_twist(np.eye(4))
        np.testing.assert_allclose(twist, np.zeros(6), atol=1e-12)

    # Pure translations -------------------------------------------------------

    def test_translation_x(self):
        twist = SE3_to_twist(sm.SE3.Tx(2.0).A)
        np.testing.assert_allclose(twist, [2, 0, 0, 0, 0, 0], atol=1e-12)

    def test_translation_y(self):
        twist = SE3_to_twist(sm.SE3.Ty(3.0).A)
        np.testing.assert_allclose(twist, [0, 3, 0, 0, 0, 0], atol=1e-12)

    def test_translation_z(self):
        twist = SE3_to_twist(sm.SE3.Tz(1.5).A)
        np.testing.assert_allclose(twist, [0, 0, 1.5, 0, 0, 0], atol=1e-12)

    def test_translation_xyz(self):
        twist = SE3_to_twist(sm.SE3.Trans(1, 2, 3).A)
        np.testing.assert_allclose(twist, [1, 2, 3, 0, 0, 0], atol=1e-12)

    # Pure rotations ----------------------------------------------------------

    def test_rotation_x_quarter(self):
        angle = np.pi / 4
        twist = SE3_to_twist(sm.SE3.Rx(angle).A)
        np.testing.assert_allclose(twist, [0, 0, 0, angle, 0, 0], atol=1e-12)

    def test_rotation_y_third(self):
        angle = np.pi / 3
        twist = SE3_to_twist(sm.SE3.Ry(angle).A)
        np.testing.assert_allclose(twist, [0, 0, 0, 0, angle, 0], atol=1e-12)

    def test_rotation_z_half(self):
        angle = np.pi / 2
        twist = SE3_to_twist(sm.SE3.Rz(angle).A)
        np.testing.assert_allclose(twist, [0, 0, 0, 0, 0, angle], atol=1e-12)

    def test_rotation_z_small(self):
        angle = 0.01
        twist = SE3_to_twist(sm.SE3.Rz(angle).A)
        np.testing.assert_allclose(twist, [0, 0, 0, 0, 0, angle], atol=1e-10)

    # Return type / shape -----------------------------------------------------

    def test_returns_ndarray(self):
        assert isinstance(SE3_to_twist(np.eye(4)), np.ndarray)

    def test_returns_shape_6(self):
        assert SE3_to_twist(np.eye(4)).shape == (6,)

    # Round-trip --------------------------------------------------------------

    @pytest.mark.parametrize("T", [
        (sm.SE3.Trans(1, 2, 0.5) * sm.SE3.Rz(0.3) * sm.SE3.Ry(0.1)).A,
        (sm.SE3.Trans(-1, 0, 3) * sm.SE3.Rx(0.5)).A,
        (_make_se3(0.5, -1.0, 2.0, 0.1, -0.2, 0.3)).A,
    ])
    def test_round_trip_to_SE3(self, T):
        twist = SE3_to_twist(T)
        T_rec = twist_to_SE3(twist)
        np.testing.assert_allclose(T_rec, T, atol=1e-10)

    def test_screw_motion_round_trip(self):
        # True screw: rotation about z + translation along z simultaneously
        T = (sm.SE3.Tz(1.0) * sm.SE3.Rz(np.pi / 4)).A
        twist = SE3_to_twist(T)
        T_back = twist_to_SE3(twist)
        np.testing.assert_allclose(T_back, T, atol=1e-10)

    def test_angular_velocity_magnitude(self):
        # The angular part of the twist must have magnitude equal to rotation angle
        angle = np.pi / 3
        twist = SE3_to_twist(sm.SE3.Rz(angle).A)
        omega = twist[3:]
        np.testing.assert_allclose(np.linalg.norm(omega), angle, atol=1e-10)

    def test_large_rotation_near_pi(self):
        # Matrix logarithm can be numerically sensitive near pi
        T = (sm.SE3.Trans(1, 2, 3) * sm.SE3.Rz(np.pi * 0.99)).A
        twist = SE3_to_twist(T)
        T_back = twist_to_SE3(twist)
        np.testing.assert_allclose(T_back, T, atol=1e-8)

    # Speed -------------------------------------------------------------------

    def test_speed_1k_calls(self):
        """1 000 calls must complete in under 500 ms."""
        T = (sm.SE3.Trans(1, 2, 3) * sm.SE3.Rz(0.5)).A
        t0 = time.perf_counter()
        for _ in range(1_000):
            SE3_to_twist(T)
        assert time.perf_counter() - t0 < 0.5


# ═════════════════════════════════════════════════════════════════════════════
# Q1.3 — twist_to_SE3
# ═════════════════════════════════════════════════════════════════════════════

class TestTwistToSE3:
    """
    Reference: Corke §2.3 — matrix exponential maps se(3) → SE(3).

    exp([0; 0; 0; 0; 0; 0]) = I₄
    exp([tx; ty; tz; 0; 0; 0]) = pure translation
    exp([0; 0; 0; ωx·θ; ωy·θ; ωz·θ]) = pure rotation
    """

    def test_zero_twist_is_identity(self):
        np.testing.assert_allclose(twist_to_SE3(np.zeros(6)), np.eye(4), atol=1e-12)

    # Pure translations -------------------------------------------------------

    def test_translation_x(self):
        expected = sm.SE3.Tx(3.0).A
        np.testing.assert_allclose(twist_to_SE3(np.array([3, 0, 0, 0, 0, 0])),
                                   expected, atol=1e-12)

    def test_translation_xyz(self):
        expected = sm.SE3.Trans(1, 2, 3).A
        np.testing.assert_allclose(twist_to_SE3(np.array([1, 2, 3, 0, 0, 0])),
                                   expected, atol=1e-12)

    # Pure rotations ----------------------------------------------------------

    def test_rotation_z_quarter(self):
        angle = np.pi / 4
        T = twist_to_SE3(np.array([0, 0, 0, 0, 0, angle]))
        np.testing.assert_allclose(T, sm.SE3.Rz(angle).A, atol=1e-12)

    def test_rotation_x(self):
        angle = np.pi / 6
        T = twist_to_SE3(np.array([0, 0, 0, angle, 0, 0]))
        np.testing.assert_allclose(T, sm.SE3.Rx(angle).A, atol=1e-12)

    # Structural validity -----------------------------------------------------

    def test_returns_4x4(self):
        assert twist_to_SE3(np.zeros(6)).shape == (4, 4)

    def test_result_is_valid_se3_matrix(self):
        twist = np.array([1.0, 0, 0, 0, 0, np.pi / 6])
        _assert_valid_se3(twist_to_SE3(twist))

    def test_zero_twist_valid_se3(self):
        _assert_valid_se3(twist_to_SE3(np.zeros(6)))

    # Round-trip --------------------------------------------------------------

    @pytest.mark.parametrize("twist", [
        np.array([0.5, -0.3, 1.2, 0.1, -0.2, 0.4]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    ])
    def test_round_trip_to_twist(self, twist):
        T = twist_to_SE3(twist)
        twist_rec = SE3_to_twist(T)
        np.testing.assert_allclose(twist_rec, twist, atol=1e-10)

    def test_screw_motion_round_trip(self):
        # True screw motion: rotation and translation along same axis
        T_orig = (sm.SE3.Tz(1.0) * sm.SE3.Rz(np.pi / 4)).A
        twist = smb.trlog(T_orig, twist=True)
        T_back = twist_to_SE3(twist)
        np.testing.assert_allclose(T_back, T_orig, atol=1e-10)

    def test_bottom_row_is_always_0001(self):
        # SE3 structural requirement: bottom row must always be [0, 0, 0, 1]
        for twist in [
            np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3]),
            np.array([0.5, -0.3, 1.2, 0.1, -0.2, 0.4]),
        ]:
            T = twist_to_SE3(twist)
            np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_large_rotation_near_pi(self):
        # Numerical stability check near the pi singularity
        twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi * 0.99])
        T = twist_to_SE3(twist)
        _assert_valid_se3(T)

    # Speed -------------------------------------------------------------------

    def test_speed_1k_calls(self):
        """1 000 calls must complete in under 500 ms."""
        twist = np.array([1, 2, 3, 0.1, 0.2, 0.3])
        t0 = time.perf_counter()
        for _ in range(1_000):
            twist_to_SE3(twist)
        assert time.perf_counter() - t0 < 0.5


# ═════════════════════════════════════════════════════════════════════════════
# Q2.1 — SE3_twist_traj
# ═════════════════════════════════════════════════════════════════════════════

class TestSE3TwistTraj:
    """
    Reference: Corke §3.1 — quintic polynomial trajectory in Cartesian space.

    Key properties of a quintic profile:
      • q(0) = q_start, q(1) = q_end     (endpoint interpolation)
      • q'(0) = q'(1) = 0                (zero velocity at endpoints)
      • q''(0) = q''(1) = 0              (zero acceleration at endpoints)
    """

    def test_length_matches_n(self, world_traj_21):
        assert len(world_traj_21) == 21

    def test_first_pose_matches_start(self, world_traj_21):
        np.testing.assert_allclose(world_traj_21[0].A, sm.SE3().A, atol=1e-10)

    def test_last_pose_matches_end(self, world_traj_21):
        expected = _make_se3(6.0, 3.0, 1.0, 0.0, 0.0, np.pi / 3).A
        np.testing.assert_allclose(world_traj_21[-1].A, expected, atol=1e-10)

    def test_returns_sm_SE3(self, world_traj_21):
        assert isinstance(world_traj_21, sm.SE3)

    def test_all_poses_valid_se3(self, world_traj_21):
        for i in range(len(world_traj_21)):
            _assert_valid_se3(world_traj_21[i].A)

    def test_identical_start_end_gives_constant_traj(self):
        """No motion requested → every pose equals the start pose."""
        pose = _make_se3(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        traj = SE3_twist_traj(pose, pose, 10)
        for i in range(len(traj)):
            np.testing.assert_allclose(traj[i].A, pose.A, atol=1e-10)

    def test_quintic_zero_velocity_at_endpoints(self):
        """
        Quintic profile property: velocity is proportional to the gap between
        consecutive poses. Near the endpoints the gap must be *much* smaller
        than near the midpoint (slug-shaped speed profile).
        """
        wTa = sm.SE3()
        wTb = _make_se3(4.0, 0.0, 0.0)
        n = 50
        traj = SE3_twist_traj(wTa, wTb, n)

        # Gap between pose 0→1 and pose (n/2-1)→(n/2)
        gap_start = np.linalg.norm(traj[1].A - traj[0].A)
        gap_mid = np.linalg.norm(traj[n // 2].A - traj[n // 2 - 1].A)
        assert gap_start < gap_mid * 0.5, (
            "Quintic profile should move slowly at the start"
        )

    def test_length_n1(self):
        wTa = sm.SE3()
        wTb = _make_se3(1.0, 0.0, 0.0)
        traj = SE3_twist_traj(wTa, wTb, 1)
        assert len(traj) == 1

    def test_n1_valid_se3(self):
        """n=1: single pose must be a valid SE3 between wTa and wTb."""
        wTa = sm.SE3()
        wTb = _make_se3(1.0, 0.0, 0.0)
        traj = SE3_twist_traj(wTa, wTb, 1)
        assert len(traj) == 1
        _assert_valid_se3(traj[0].A)

    def test_non_identity_start_intermediate(self):
        """Non-identity wTa: intermediate poses must lie on the SE3 geodesic."""
        wTa = _make_se3(1.0, 2.0, 0.5, 0.0, 0.0, 0.3)
        wTb = _make_se3(4.0, -1.0, 2.0, 0.0, 0.0, 1.1)
        n = 9
        traj = SE3_twist_traj(wTa, wTb, n)
        assert len(traj) == n
        np.testing.assert_allclose(traj[0].A, wTa.A, atol=1e-10)
        np.testing.assert_allclose(traj[-1].A, wTb.A, atol=1e-10)
        for i in range(n):
            _assert_valid_se3(traj[i].A)

    def test_combined_rotation_translation(self):
        """Combined rotation+translation: endpoints exact, intermediates valid SE3."""
        wTa = _make_se3(0.5, -1.0, 1.5, 0.2, -0.3, 0.7)
        wTb = _make_se3(3.0,  2.0, 0.0, -0.1, 0.4, -0.5)
        n = 11
        traj = SE3_twist_traj(wTa, wTb, n)
        assert len(traj) == n
        np.testing.assert_allclose(traj[0].A, wTa.A, atol=1e-10)
        np.testing.assert_allclose(traj[-1].A, wTb.A, atol=1e-10)
        for i in range(n):
            _assert_valid_se3(traj[i].A, atol=1e-8)

    def test_pure_translation_endpoints(self):
        wTa = sm.SE3.Tx(0.0)
        wTb = sm.SE3.Tx(5.0)
        traj = SE3_twist_traj(wTa, wTb, 20)
        np.testing.assert_allclose(traj[0].A, wTa.A, atol=1e-10)
        np.testing.assert_allclose(traj[-1].A, wTb.A, atol=1e-10)

    def test_pure_rotation_endpoints(self):
        wTa = sm.SE3()
        wTb = sm.SE3.Rz(np.pi / 2)
        traj = SE3_twist_traj(wTa, wTb, 15)
        np.testing.assert_allclose(traj[0].A, wTa.A, atol=1e-10)
        np.testing.assert_allclose(traj[-1].A, wTb.A, atol=1e-10)

    def test_n2_gives_start_and_end(self):
        # n=2 edge case: only start and end, no intermediate poses
        wTa = sm.SE3()
        wTb = _make_se3(1.0, 0.0, 0.0)
        traj = SE3_twist_traj(wTa, wTb, 2)
        assert len(traj) == 2
        np.testing.assert_allclose(traj[0].A, wTa.A, atol=1e-10)
        np.testing.assert_allclose(traj[1].A, wTb.A, atol=1e-10)

    def test_pure_translation_monotonic(self):
        # For a straight-line translation along x, x-coordinate must be
        # monotonically increasing throughout the trajectory
        wTa = sm.SE3()
        wTb = sm.SE3.Tx(5.0)
        traj = SE3_twist_traj(wTa, wTb, 20)
        x_coords = [traj[i].A[0, 3] for i in range(len(traj))]
        for i in range(1, len(x_coords)):
            assert x_coords[i] >= x_coords[i - 1] - 1e-10, (
                f"x not monotonic at index {i}: {x_coords[i-1]} → {x_coords[i]}"
            )

    def test_large_rotation_near_pi(self):
        # Numerical stability near pi rotation
        wTa = sm.SE3()
        wTb = sm.SE3.Rz(np.pi * 0.99)
        traj = SE3_twist_traj(wTa, wTb, 10)
        np.testing.assert_allclose(traj[0].A, wTa.A, atol=1e-8)
        np.testing.assert_allclose(traj[-1].A, wTb.A, atol=1e-8)
        for i in range(len(traj)):
            _assert_valid_se3(traj[i].A)

    # Speed -------------------------------------------------------------------

    def test_speed_n50(self):
        """A 50-step trajectory must be computed in under 1 s."""
        wTa = sm.SE3()
        wTb = _make_se3(1.0, 1.0, 1.0, 0.0, 0.0, np.pi / 3)
        t0 = time.perf_counter()
        SE3_twist_traj(wTa, wTb, 50)
        assert time.perf_counter() - t0 < 1.0

    def test_speed_n200(self):
        """A 200-step trajectory must be computed in under 3 s."""
        wTa = sm.SE3()
        wTb = _make_se3(2.0, 1.0, 0.5, 0.1, 0.2, 0.8)
        t0 = time.perf_counter()
        SE3_twist_traj(wTa, wTb, 200)
        assert time.perf_counter() - t0 < 3.0


# ═════════════════════════════════════════════════════════════════════════════
# Q2.2 — SE3_traj_relative
# ═════════════════════════════════════════════════════════════════════════════

class TestSE3TrajRelative:
    """
    Reference: Corke §2.1 — relative pose composition.

    rel[i] = T[i-1]⁻¹ · T[i]
    Reconstruction: T[k] = T[0] · rel[0] · rel[1] · … · rel[k-1]
    """

    def test_output_length_is_n_minus_1(self, world_traj_21):
        rel = SE3_traj_relative(world_traj_21)
        assert len(rel) == len(world_traj_21) - 1

    def test_returns_sm_SE3(self, world_traj_21):
        assert isinstance(SE3_traj_relative(world_traj_21), sm.SE3)

    def test_reconstruct_full_trajectory(self, world_traj_21):
        """Composing relative poses forward must recover every world pose."""
        rel = SE3_traj_relative(world_traj_21)
        current = world_traj_21[0]
        for i in range(len(rel)):
            current = current * rel[i]
            np.testing.assert_allclose(
                current.A, world_traj_21[i + 1].A, atol=1e-8,
                err_msg=f"Mismatch at index {i + 1}",
            )

    def test_constant_world_traj_gives_identity_relatives(self):
        """Stationary trajectory: all relative poses must be identity."""
        pose = _make_se3(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        traj = sm.SE3.Empty()
        for _ in range(6):
            traj.append(pose)
        rel = SE3_traj_relative(traj)
        for i in range(len(rel)):
            np.testing.assert_allclose(rel[i].A, np.eye(4), atol=1e-10)

    def test_two_pose_traj_gives_one_relative(self):
        wTa = sm.SE3()
        wTb = _make_se3(1.0, 0.0, 0.0)
        traj = sm.SE3.Empty()
        traj.append(wTa)
        traj.append(wTb)
        rel = SE3_traj_relative(traj)
        assert len(rel) == 1
        np.testing.assert_allclose(rel[0].A, wTb.A, atol=1e-10)

    def test_all_relatives_are_valid_se3(self, world_traj_21):
        rel = SE3_traj_relative(world_traj_21)
        for i in range(len(rel)):
            _assert_valid_se3(rel[i].A)

    def test_pure_translation_step_size(self):
        """Each relative pose in a straight-line traj should be a fixed x-step."""
        step = sm.SE3.Tx(0.5)
        traj = sm.SE3.Empty()
        current = sm.SE3()
        for _ in range(6):
            traj.append(current)
            current = current * step
        rel = SE3_traj_relative(traj)
        for i in range(len(rel)):
            np.testing.assert_allclose(rel[i].A, step.A, atol=1e-10)

    # Speed -------------------------------------------------------------------

    def test_speed_200_pose_traj(self):
        """Converting a 200-pose trajectory must complete in under 1 s."""
        wTa = sm.SE3()
        wTb = _make_se3(5.0, 5.0, 5.0, 0.5, 0.5, 0.5)
        traj = SE3_twist_traj(wTa, wTb, 200)
        t0 = time.perf_counter()
        SE3_traj_relative(traj)
        assert time.perf_counter() - t0 < 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Q3.1 — quintic_properties_symbolic
# ═════════════════════════════════════════════════════════════════════════════

class TestQuinticPropertiesSymbolic:
    """
    Reference: Corke §3.1, Lynch & Park §9.2 — quintic trajectory properties.

    Standard unit-step quintic: q(t) = 6t⁵ − 15t⁴ + 10t³
      Coefficients (A, B, C, D, E, F) = (6, −15, 10, 0, 0, 0)

    Analytically (via matrix exponential / closed-form roots):
      q''(t) = 120t³ − 180t² + 60t = 60t(2t−1)(t−1)
      Critical points of |q''|: solve q'''=0 → t = (3 ± √3)/6
      max |q''| = q''((3−√3)/6) = 10/√3 = 10√3/3 ≈ 5.7735

      q'''(t) = 360t² − 360t + 60
      Critical point of |q'''|: solve q''''=0 → t = 1/2
      q'''(0) = q'''(1) = 60, q'''(1/2) = −30  → max |q'''| = 60
    """

    STANDARD = (6, -15, 10, 0, 0, 0)
    MAX_ACC_STANDARD = 10.0 * np.sqrt(3) / 3      # ≈ 5.7735
    MAX_JERK_STANDARD = 60.0

    # Standard quintic --------------------------------------------------------

    def test_standard_max_acc(self):
        acc, _ = quintic_properties_symbolic(self.STANDARD)
        np.testing.assert_allclose(acc, self.MAX_ACC_STANDARD, rtol=1e-6)

    def test_standard_max_jerk(self):
        _, jerk = quintic_properties_symbolic(self.STANDARD)
        np.testing.assert_allclose(jerk, self.MAX_JERK_STANDARD, rtol=1e-6)

    def test_standard_returns_tuple_of_two(self):
        result = quintic_properties_symbolic(self.STANDARD)
        assert len(result) == 2

    def test_standard_values_are_float(self):
        acc, jerk = quintic_properties_symbolic(self.STANDARD)
        assert isinstance(acc, float)
        assert isinstance(jerk, float)

    # Degenerate / simple polynomials ----------------------------------------

    def test_zero_polynomial(self):
        """All-zero coefficients → zero acceleration and jerk everywhere."""
        acc, jerk = quintic_properties_symbolic((0, 0, 0, 0, 0, 0))
        assert acc == pytest.approx(0.0)
        assert jerk == pytest.approx(0.0)

    def test_constant_polynomial(self):
        """q(t) = F (constant) → zero derivatives."""
        acc, jerk = quintic_properties_symbolic((0, 0, 0, 0, 0, 5))
        assert acc == pytest.approx(0.0)
        assert jerk == pytest.approx(0.0)

    def test_linear_polynomial(self):
        """q(t) = t → q'' = 0, q''' = 0."""
        acc, jerk = quintic_properties_symbolic((0, 0, 0, 0, 1, 0))
        assert acc == pytest.approx(0.0)
        assert jerk == pytest.approx(0.0)

    def test_quadratic_polynomial(self):
        """q(t) = t² → q'' = 2 (constant), q''' = 0."""
        acc, jerk = quintic_properties_symbolic((0, 0, 0, 1, 0, 0))
        assert acc == pytest.approx(2.0)
        assert jerk == pytest.approx(0.0)

    def test_cubic_polynomial(self):
        """
        q(t) = t³ → q'' = 6t (max at t=1 → 6), q''' = 6 (constant → 6).
        """
        acc, jerk = quintic_properties_symbolic((0, 0, 1, 0, 0, 0))
        assert acc == pytest.approx(6.0)
        assert jerk == pytest.approx(6.0)

    def test_quartic_polynomial(self):
        """
        q(t) = t⁴ → q'' = 12t² (max at t=1 → 12), q''' = 24t (max at t=1 → 24).
        """
        acc, jerk = quintic_properties_symbolic((0, 1, 0, 0, 0, 0))
        assert acc == pytest.approx(12.0)
        assert jerk == pytest.approx(24.0)

    # Sign invariance ---------------------------------------------------------

    def test_negated_coefficients_same_magnitudes(self):
        """Negating all coefficients must not change max absolute values."""
        neg = tuple(-c for c in self.STANDARD)
        acc1, jerk1 = quintic_properties_symbolic(self.STANDARD)
        acc2, jerk2 = quintic_properties_symbolic(neg)
        assert acc1 == pytest.approx(acc2, rel=1e-6)
        assert jerk1 == pytest.approx(jerk2, rel=1e-6)

    def test_jerk_maximum_at_interior_point(self):
        # For the standard quintic, jerk maximum at endpoints (t=0, t=1) = 60
        # and interior minimum at t=0.5 = -30, confirming interior critical point exists
        _, jerk = quintic_properties_symbolic(self.STANDARD)
        # max |jerk| must be at endpoints, not interior
        np.testing.assert_allclose(jerk, 60.0, rtol=1e-6)

    def test_acc_maximum_at_interior_point(self):
        # For standard quintic, max |acc| occurs strictly inside (0,1)
        # at t=(3±√3)/6 ≈ 0.211 and 0.789, NOT at endpoints where acc=0
        acc, _ = quintic_properties_symbolic(self.STANDARD)
        # Endpoints give acc=0, so any positive result confirms interior maximum
        assert acc > 0.1, "Maximum acceleration should occur at an interior point"
        np.testing.assert_allclose(acc, 10.0 * np.sqrt(3) / 3, rtol=1e-6)

    def test_quintic_with_nonzero_boundary_conditions(self):
        # Non-standard quintic with D != 0 (nonzero initial acceleration)
        # q(t) = t^5 - 2t^4 + t^2 → acc = 20t^3 - 24t^2 + 2, jerk = 60t^2 - 48t
        acc, jerk = quintic_properties_symbolic((1, -2, 0, 1, 0, 0))
        assert acc >= 0.0
        assert jerk >= 0.0
        # Verify against numpy reference
        acc_poly  = [20, -24, 0, 2]
        jerk_poly = [60, -48, 0]
        roots = np.roots([60, -48, 0])  # derivative of jerk
        pts = [0.0, 1.0] + [r.real for r in roots
                            if abs(r.imag) < 1e-10 and 0.0 <= r.real <= 1.0]
        expected_jerk = float(max(abs(np.polyval(jerk_poly, t)) for t in pts))
        np.testing.assert_allclose(jerk, expected_jerk, rtol=1e-5)

    # Non-negativity ----------------------------------------------------------

    def test_max_acc_non_negative(self):
        acc, _ = quintic_properties_symbolic((1, -2, 3, -1, 0.5, 0))
        assert acc >= 0.0

    def test_max_jerk_non_negative(self):
        _, jerk = quintic_properties_symbolic((1, -2, 3, -1, 0.5, 0))
        assert jerk >= 0.0

    # Speed -------------------------------------------------------------------

    def test_speed_single_call(self):
        """One symbolic evaluation must complete in under 5 s."""
        t0 = time.perf_counter()
        quintic_properties_symbolic(self.STANDARD)
        assert time.perf_counter() - t0 < 5.0

    def test_speed_five_distinct_calls(self):
        """Five calls with different polynomials must finish in under 20 s."""
        cases = [
            (6, -15, 10, 0, 0, 0),
            (0, 0, 0, 1, 0, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 1, 0, 0, 0, 0),
            (1, -1, 1, -1, 0, 0),
        ]
        t0 = time.perf_counter()
        for c in cases:
            quintic_properties_symbolic(c)
        assert time.perf_counter() - t0 < 20.0


# ═════════════════════════════════════════════════════════════════════════════
# Q4.1 — rocket_diagnostics
# ═════════════════════════════════════════════════════════════════════════════

class TestRocketDiagnostics:
    """
    The relative trajectory is created from a known world-frame trajectory so
    the expected time at each world pose is exactly idx × 0.5 s.
    """

    def test_finds_start_pose_at_t0(self, rel_traj_10, world_traj_11):
        """World origin (identity) is the start; function should return 0.0 s."""
        t = rocket_diagnostics(rel_traj_10, world_traj_11[0])
        assert t == pytest.approx(0.0)

    def test_finds_final_pose(self, rel_traj_10, world_traj_11):
        """Last world pose is reached after all 10 relative steps → t = 5.0 s."""
        t = rocket_diagnostics(rel_traj_10, world_traj_11[-1])
        assert t == pytest.approx(5.0)

    def test_finds_midpoint_pose(self, rel_traj_10, world_traj_11):
        """world_traj_11[5] is reached after 5 relative steps → t = 2.5 s."""
        t = rocket_diagnostics(rel_traj_10, world_traj_11[5])
        assert t == pytest.approx(2.5)

    def test_all_indices_correct_timing(self, rel_traj_10, world_traj_11):
        """Every world pose index k should be found at t = k × 0.5 s."""
        for idx in range(len(world_traj_11)):
            t = rocket_diagnostics(rel_traj_10, world_traj_11[idx])
            assert t == pytest.approx(idx * 0.5), (
                f"index {idx}: expected {idx * 0.5} s, got {t} s"
            )

    def test_pure_translation_single_step(self):
        """Each step is Tx(1) → after 3 steps world pose is Tx(3) at t=1.5 s."""
        step = sm.SE3.Tx(1.0)
        traj = sm.SE3.Empty()
        traj.append(sm.SE3())  # world[0] = identity (prepended start)
        for _ in range(5):
            traj.append(step)
        assert rocket_diagnostics(traj, sm.SE3.Tx(3.0)) == pytest.approx(1.5)

    def test_pure_translation_first_step(self):
        step = sm.SE3.Tx(2.0)
        traj = sm.SE3.Empty()
        traj.append(sm.SE3())  # world[0] = identity (prepended start)
        traj.append(step)
        assert rocket_diagnostics(traj, sm.SE3.Tx(2.0)) == pytest.approx(0.5)

    def test_no_match_returns_none(self):
        """A pose not in the trajectory should return None (graceful miss)."""
        step = sm.SE3.Tx(1.0)
        traj = sm.SE3.Empty()
        traj.append(sm.SE3())  # world[0] = identity (prepended start)
        for _ in range(3):
            traj.append(step)
        result = rocket_diagnostics(traj, sm.SE3.Ty(999.0))
        assert result is None

    def test_non_identity_start_pose(self):
        # Rocket starts at a non-identity world pose — accumulation must still work
        start = _make_se3(1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 6)
        end   = _make_se3(4.0, 5.0, 6.0, 0.0, 0.0, np.pi / 3)
        world = SE3_twist_traj(start, end, 7)
        rel   = SE3_traj_relative(world)

        # Prepend world[0] as first element (autograder structure)
        full_rel = sm.SE3.Empty()
        full_rel.append(start)  # world[0] = start
        for i in range(len(rel)):
            full_rel.append(rel[i])

        t = rocket_diagnostics(full_rel, world[-1])
        # world[-1] = world[6], reached after applying 6 elements → t = 6*0.5 = 3.0
        assert t == pytest.approx((len(world) - 1) * 0.5, abs=1e-6)

    def test_tolerance_boundary_inside(self):
        # A pose perturbed by less than 1e-4 should still match
        step = sm.SE3.Tx(1.0)
        traj = sm.SE3.Empty()
        traj.append(sm.SE3())  # world[0] = identity (prepended start)
        for _ in range(4):
            traj.append(step)
        # Perturb only the translation element — keeps matrix structure valid
        M = sm.SE3.Tx(2.0).A.copy()
        M[0, 3] += 5e-5  # nudge x-translation by 5e-5, within 1e-4 tolerance
        T_perturbed = sm.SE3(M, check=False)
        t = rocket_diagnostics(traj, T_perturbed)
        assert t == pytest.approx(2 * 0.5)

    def test_tolerance_boundary_outside(self):
        # A pose perturbed by more than 1e-4 should NOT match → returns None
        step = sm.SE3.Tx(1.0)
        traj = sm.SE3.Empty()
        traj.append(sm.SE3())  # world[0] = identity (prepended start)
        for _ in range(4):
            traj.append(step)
        # Perturb only the translation element by 2e-4 — outside tolerance
        M = sm.SE3.Tx(2.0).A.copy()
        M[0, 3] += 2e-4
        T_perturbed = sm.SE3(M, check=False)
        result = rocket_diagnostics(traj, T_perturbed)
        assert result is None

    # Speed -------------------------------------------------------------------

    def test_speed_100_step_traj(self):
        """Searching a 100-step relative traj must complete in under 200 ms."""
        wTa = sm.SE3()
        wTb = _make_se3(10.0, 5.0, 2.0, 0.0, 0.0, np.pi / 3)
        world = SE3_twist_traj(wTa, wTb, 101)
        rel = SE3_traj_relative(world)
        traj = sm.SE3.Empty()
        traj.append(world[0])
        for r in rel:
            traj.append(r)
        T_goal = world[-1]
        t0 = time.perf_counter()
        rocket_diagnostics(traj, T_goal)
        assert time.perf_counter() - t0 < 0.2


# ═════════════════════════════════════════════════════════════════════════════
# Q4.2 — rocket_pose
# ═════════════════════════════════════════════════════════════════════════════

class TestRocketPose:
    """
    world_traj_21 has 21 poses, dt=0.5 s, total duration = 10 s.

    For any index a_idx:
      aTg  = traj[a_idx]⁻¹ · traj[-1]
      time = (21 − 1 − a_idx) × 0.5 s
    """

    def test_from_first_pose_full_duration(self, world_traj_21):
        aTg, t = rocket_pose(world_traj_21, world_traj_21[0])
        assert t == pytest.approx(10.0)
        expected = world_traj_21[0].inv() * world_traj_21[-1]
        np.testing.assert_allclose(aTg.A, expected.A, atol=1e-8)

    def test_from_last_pose_zero_duration(self, world_traj_21):
        aTg, t = rocket_pose(world_traj_21, world_traj_21[-1])
        assert t == pytest.approx(0.0)
        # At the goal, aTg must be identity
        np.testing.assert_allclose(aTg.A, np.eye(4), atol=1e-8)

    def test_from_midpoint(self, world_traj_21):
        mid = world_traj_21[10]
        aTg, t = rocket_pose(world_traj_21, mid)
        assert t == pytest.approx(5.0)
        expected = mid.inv() * world_traj_21[-1]
        np.testing.assert_allclose(aTg.A, expected.A, atol=1e-8)

    def test_all_indices_time_remaining(self, world_traj_21):
        """For every index the remaining time = (n-1-idx) × 0.5."""
        n = len(world_traj_21)  # 21
        for idx in range(n):
            _, t = rocket_pose(world_traj_21, world_traj_21[idx])
            expected_t = (n - 1 - idx) * 0.5
            assert t == pytest.approx(expected_t), (
                f"index {idx}: expected {expected_t} s, got {t} s"
            )

    def test_elapsed_plus_remaining_equals_total(self, world_traj_21):
        """Elapsed + remaining must always sum to total trajectory duration."""
        n = len(world_traj_21)
        total = (n - 1) * 0.5  # 10.0 s
        for idx in range(n):
            _, t_rem = rocket_pose(world_traj_21, world_traj_21[idx])
            t_elapsed = idx * 0.5
            assert t_elapsed + t_rem == pytest.approx(total)

    def test_aTg_is_sm_SE3(self, world_traj_21):
        aTg, _ = rocket_pose(world_traj_21, world_traj_21[0])
        assert isinstance(aTg, sm.SE3)

    def test_aTg_is_valid_se3_matrix(self, world_traj_21):
        aTg, _ = rocket_pose(world_traj_21, world_traj_21[5])
        _assert_valid_se3(aTg.A)

    def test_aTg_composition_recovers_goal(self, world_traj_21):
        """wTa · aTg must equal wTg (the goal in the world frame)."""
        for idx in [0, 5, 10, 15, 20]:
            wTa = world_traj_21[idx]
            aTg, _ = rocket_pose(world_traj_21, wTa)
            wTg_computed = wTa * aTg
            np.testing.assert_allclose(
                wTg_computed.A, world_traj_21[-1].A, atol=1e-8,
                err_msg=f"Composition failed at index {idx}",
            )

    def test_composition_at_goal_boundary(self, world_traj_21):
        # At the goal itself, wTa * aTg must still equal wTg
        wTa = world_traj_21[-1]
        aTg, t = rocket_pose(world_traj_21, wTa)
        wTg_computed = wTa * aTg
        np.testing.assert_allclose(
            wTg_computed.A, world_traj_21[-1].A, atol=1e-8,
            err_msg="Composition failed at goal boundary"
        )
        assert t == pytest.approx(0.0)

    def test_aTg_valid_se3_all_indices(self, world_traj_21):
        # Every aTg at every index must be a valid SE3 matrix
        for idx in range(len(world_traj_21)):
            aTg, _ = rocket_pose(world_traj_21, world_traj_21[idx])
            _assert_valid_se3(aTg.A, atol=1e-7)

    def test_pure_translation_trajectory(self):
        """Simple straight-line: each world pose is Tx(k)."""
        traj = sm.SE3.Empty()
        for k in range(6):
            traj.append(sm.SE3.Tx(float(k)))
        wTa = sm.SE3.Tx(2.0)
        aTg, t = rocket_pose(traj, wTa)
        # Remaining steps: 5 - 2 = 3, time = 3 × 0.5 = 1.5 s
        assert t == pytest.approx(1.5)
        expected_aTg = wTa.inv() * sm.SE3.Tx(5.0)
        np.testing.assert_allclose(aTg.A, expected_aTg.A, atol=1e-8)

    # Speed -------------------------------------------------------------------

    def test_speed_traj_100(self):
        """rocket_pose on a 101-pose trajectory must complete in under 200 ms."""
        wTa_s = sm.SE3()
        wTb_e = _make_se3(10.0, 5.0, 2.0, 0.0, 0.0, 1.0)
        traj = SE3_twist_traj(wTa_s, wTb_e, 101)
        pose = traj[50]
        t0 = time.perf_counter()
        rocket_pose(traj, pose)
        assert time.perf_counter() - t0 < 0.2
