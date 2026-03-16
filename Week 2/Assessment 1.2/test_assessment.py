import numpy as np
import pytest
from orientation import (
    is_SO2, is_SO3, RPY_to_SO3,
    is_eulvec, is_angax, angax_to_eulvec, eulvec_to_SO3,
    is_quat, quat_to_SO3, SO3_to_quat,
    orientation_type, filter_orientations, get_planet_orienations
)

# ================================================================
# Helpers
# ================================================================

def rot2d(theta):
    """Valid SO(2) rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def rot3d_z(theta):
    """Valid SO(3) rotation about z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def rot3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def rot3d_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

# ================================================================
# Question 1.1 — is_SO2
# ================================================================

class TestIsSO2:
    def test_identity(self):
        assert is_SO2(np.eye(2)) is True

    def test_valid_rotation(self):
        assert is_SO2(rot2d(np.pi / 4)) is True

    def test_valid_rotation_180(self):
        assert is_SO2(rot2d(np.pi)) is True

    def test_wrong_shape_3x3(self):
        assert is_SO2(np.eye(3)) is False

    def test_wrong_shape_1d(self):
        assert is_SO2(np.array([1, 0])) is False

    def test_not_orthogonal(self):
        M = np.array([[2.0, 0.0], [0.0, 0.5]])
        assert is_SO2(M) is False

    def test_det_minus_one(self):
        # Orthogonal but det = -1 (reflection)
        M = np.array([[1.0, 0.0], [0.0, -1.0]])
        assert is_SO2(M) is False

    def test_near_valid(self):
        # Slightly perturbed — should still pass within tol
        M = rot2d(1.0) + np.eye(2) * 1e-8
        # Not guaranteed true, just shouldn't crash
        result = is_SO2(M)
        assert isinstance(result, bool)

    def test_zero_matrix(self):
        assert is_SO2(np.zeros((2, 2))) is False


# ================================================================
# Question 1.2 — is_SO3
# ================================================================

class TestIsSO3:
    def test_identity(self):
        assert is_SO3(np.eye(3)) is True

    def test_valid_rotation_z(self):
        assert is_SO3(rot3d_z(np.pi / 3)) is True

    def test_valid_rotation_x(self):
        assert is_SO3(rot3d_x(np.pi / 5)) is True

    def test_valid_combined(self):
        R = rot3d_z(0.3) @ rot3d_y(0.5) @ rot3d_x(0.7)
        assert is_SO3(R) is True

    def test_wrong_shape_2x2(self):
        assert is_SO3(np.eye(2)) is False

    def test_wrong_shape_4x4(self):
        assert is_SO3(np.eye(4)) is False

    def test_not_orthogonal(self):
        M = np.diag([2.0, 0.5, 1.0])
        assert is_SO3(M) is False

    def test_det_minus_one(self):
        M = np.diag([1.0, 1.0, -1.0])  # reflection
        assert is_SO3(M) is False

    def test_zero_matrix(self):
        assert is_SO3(np.zeros((3, 3))) is False


# ================================================================
# Question 1.3 — RPY_to_SO3
# ================================================================

class TestRPYtoSO3:
    def test_zero_angles(self):
        rpy = np.array([[0.0], [0.0], [0.0]])
        R = RPY_to_SO3(rpy)
        assert np.allclose(R, np.eye(3), atol=1e-6)

    def test_returns_SO3(self):
        rpy = np.array([[0.1], [0.2], [0.3]])
        R = RPY_to_SO3(rpy)
        assert is_SO3(R)

    def test_yaw_only(self):
        """Pure yaw = rotation about z."""
        yaw = np.pi / 4
        rpy = np.array([[0.0], [0.0], [yaw]])
        R = RPY_to_SO3(rpy)
        assert np.allclose(R, rot3d_z(yaw), atol=1e-6)

    def test_pitch_only(self):
        """Pure pitch = rotation about y."""
        pitch = np.pi / 6
        rpy = np.array([[0.0], [pitch], [0.0]])
        R = RPY_to_SO3(rpy)
        assert np.allclose(R, rot3d_y(pitch), atol=1e-6)

    def test_roll_only(self):
        """Pure roll = rotation about x."""
        roll = np.pi / 3
        rpy = np.array([[roll], [0.0], [0.0]])
        R = RPY_to_SO3(rpy)
        assert np.allclose(R, rot3d_x(roll), atol=1e-6)

    def test_combined_angles(self):
        roll, pitch, yaw = 0.1, 0.2, 0.3
        rpy = np.array([[roll], [pitch], [yaw]])
        R = RPY_to_SO3(rpy)
        R_expected = rot3d_z(yaw) @ rot3d_y(pitch) @ rot3d_x(roll)  # fixed
        assert np.allclose(R, R_expected, atol=1e-6)

    def test_output_shape(self):
        rpy = np.array([[0.1], [0.2], [0.3]])
        assert RPY_to_SO3(rpy).shape == (3, 3)


# ================================================================
# Question 2.1 — is_eulvec
# ================================================================

class TestIsEulvec:
    def test_valid(self):
        v = np.array([[0.1], [0.2], [0.3]])
        assert is_eulvec(v) is True

    def test_zero_vector(self):
        # Zero Euler vector = identity rotation — should be valid
        v = np.zeros((3, 1))
        assert is_eulvec(v) is True

    def test_wrong_shape_4x1(self):
        v = np.array([[0.1], [0.2], [0.3], [0.4]])
        assert is_eulvec(v) is False

    def test_wrong_shape_1d(self):
        v = np.array([0.1, 0.2, 0.3])
        assert is_eulvec(v) is False

    def test_wrong_shape_3x3(self):
        assert is_eulvec(np.eye(3)) is False


# ================================================================
# Question 2.2 — is_angax
# ================================================================

class TestIsAngax:
    def test_valid(self):
        axis = np.array([0.0, 0.0, 1.0])
        v = np.array([[np.pi / 4], [axis[0]], [axis[1]], [axis[2]]])
        assert is_angax(v) is True

    def test_axis_not_unit(self):
        v = np.array([[np.pi / 4], [1.0], [1.0], [0.0]])  # not normalised
        assert is_angax(v) is False

    def test_wrong_shape(self):
        v = np.array([[0.1], [0.0], [0.0]])
        assert is_angax(v) is False

    def test_zero_axis(self):
        v = np.array([[np.pi / 4], [0.0], [0.0], [0.0]])
        assert is_angax(v) is False

    def test_identity_rotation(self):
        # theta=0, any unit axis should be valid
        v = np.array([[0.0], [1.0], [0.0], [0.0]])
        assert is_angax(v) is True


# ================================================================
# Question 2.3 — angax_to_eulvec
# ================================================================

class TestAngaxToEulvec:
    def test_basic(self):
        theta = np.pi / 3
        axis = np.array([0.0, 0.0, 1.0])
        angax = np.array([[theta], [axis[0]], [axis[1]], [axis[2]]])
        ev = angax_to_eulvec(angax)
        expected = np.array([[axis[0] * theta], [axis[1] * theta], [axis[2] * theta]])
        assert np.allclose(ev, expected, atol=1e-6)

    def test_output_shape(self):
        angax = np.array([[1.0], [1.0], [0.0], [0.0]])
        ev = angax_to_eulvec(angax)
        assert ev.shape == (3, 1)

    def test_zero_angle(self):
        angax = np.array([[0.0], [1.0], [0.0], [0.0]])
        ev = angax_to_eulvec(angax)
        assert np.allclose(ev, np.zeros((3, 1)), atol=1e-6)

    def test_is_eulvec(self):
        angax = np.array([[0.5], [0.0], [1.0], [0.0]])
        ev = angax_to_eulvec(angax)
        assert is_eulvec(ev)


# ================================================================
# Question 2.4 — eulvec_to_SO3
# ================================================================

class TestEulvecToSO3:
    def test_zero_is_identity(self):
        ev = np.zeros((3, 1))
        R = eulvec_to_SO3(ev)
        assert np.allclose(R, np.eye(3), atol=1e-6)

    def test_returns_SO3(self):
        ev = np.array([[0.1], [0.2], [0.3]])
        assert is_SO3(eulvec_to_SO3(ev))

    def test_rotation_about_z(self):
        theta = np.pi / 4
        ev = np.array([[0.0], [0.0], [theta]])
        R = eulvec_to_SO3(ev)
        assert np.allclose(R, rot3d_z(theta), atol=1e-6)

    def test_output_shape(self):
        ev = np.array([[0.1], [0.2], [0.3]])
        assert eulvec_to_SO3(ev).shape == (3, 3)


# ================================================================
# Question 3.1 — is_quat
# ================================================================

class TestIsQuat:
    def test_identity_quat(self):
        q = np.array([[1.0], [0.0], [0.0], [0.0]])
        assert is_quat(q) is True

    def test_valid_quat(self):
        q = np.array([[np.cos(0.5)], [np.sin(0.5)], [0.0], [0.0]])
        assert is_quat(q) is True

    def test_not_unit(self):
        q = np.array([[2.0], [0.0], [0.0], [0.0]])
        assert is_quat(q) is False

    def test_wrong_shape(self):
        q = np.array([[1.0], [0.0], [0.0]])
        assert is_quat(q) is False

    def test_zero_quat(self):
        assert is_quat(np.zeros((4, 1))) is False

    def test_1d_array(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        assert is_quat(q) is False


# ================================================================
# Question 3.2 — quat_to_SO3
# ================================================================

class TestQuatToSO3:
    def test_identity(self):
        q = np.array([[1.0], [0.0], [0.0], [0.0]])
        assert np.allclose(quat_to_SO3(q), np.eye(3), atol=1e-6)

    def test_returns_SO3(self):
        theta = np.pi / 4
        q = np.array([[np.cos(theta/2)], [0.0], [0.0], [np.sin(theta/2)]])
        assert is_SO3(quat_to_SO3(q))

    def test_rotation_about_z(self):
        theta = np.pi / 3
        q = np.array([[np.cos(theta/2)], [0.0], [0.0], [np.sin(theta/2)]])
        R = quat_to_SO3(q)
        assert np.allclose(R, rot3d_z(theta), atol=1e-6)

    def test_output_shape(self):
        q = np.array([[1.0], [0.0], [0.0], [0.0]])
        assert quat_to_SO3(q).shape == (3, 3)


# ================================================================
# Question 3.3 — SO3_to_quat
# ================================================================

class TestSO3ToQuat:
    def test_identity(self):
        q = SO3_to_quat(np.eye(3))
        # Either [1,0,0,0] or [-1,0,0,0] are valid
        assert np.allclose(np.abs(q[0]), 1.0, atol=1e-6)
        assert np.allclose(q[1:], 0.0, atol=1e-6)

    def test_output_shape(self):
        assert SO3_to_quat(np.eye(3)).shape == (4, 1)

    def test_is_unit_quat(self):
        R = rot3d_z(np.pi / 5)
        q = SO3_to_quat(R)
        assert is_quat(q)

    def test_roundtrip(self):
        """SO3 -> quat -> SO3 should recover original."""
        R = rot3d_z(0.3) @ rot3d_y(0.5) @ rot3d_x(0.7)
        q = SO3_to_quat(R)
        R2 = quat_to_SO3(q)
        assert np.allclose(R, R2, atol=1e-6)

    def test_rotation_about_z(self):
        theta = np.pi / 3
        R = rot3d_z(theta)
        q = SO3_to_quat(R)
        q_expected = np.array([[np.cos(theta/2)], [0.0], [0.0], [np.sin(theta/2)]])
        # Allow sign flip
        assert np.allclose(q, q_expected, atol=1e-6) or np.allclose(q, -q_expected, atol=1e-6)


# ================================================================
# Question 4.1 — orientation_type
# ================================================================

class TestOrientationType:
    def test_SO3(self):
        assert orientation_type(np.eye(3)) == "SO3"

    def test_quat(self):
        q = np.array([[1.0], [0.0], [0.0], [0.0]])
        assert orientation_type(q) == "quat"

    def test_eulvec(self):
        v = np.array([[0.1], [0.2], [0.3]])
        assert orientation_type(v) == "eulvec"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            orientation_type(np.zeros((5, 1)))

    def test_invalid_matrix_raises(self):
        with pytest.raises(ValueError):
            orientation_type(np.zeros((3, 3)))  # not a valid SO3


# ================================================================
# Question 4.2 — filter_orientations
# ================================================================

class TestFilterOrientations:
    def _make_mixed_list(self):
        R = rot3d_z(0.3)
        q = np.array([[1.0], [0.0], [0.0], [0.0]])
        ev = np.array([[0.1], [0.2], [0.3]])
        bad = np.zeros((5, 1))
        return [R, q, ev, bad]

    def test_removes_errors(self):
        result = filter_orientations(self._make_mixed_list())
        assert len(result) == 3

    def test_all_SO3(self):
        result = filter_orientations(self._make_mixed_list())
        assert all(is_SO3(r) for r in result)

    def test_consistent_type(self):
        result = filter_orientations(self._make_mixed_list())
        assert all(r.shape == (3, 3) for r in result)

    def test_empty_list(self):
        assert filter_orientations([]) == []

    def test_all_invalid(self):
        bad = [np.zeros((5, 1)), np.zeros((2, 2))]
        assert filter_orientations(bad) == []

    def test_all_already_SO3(self):
        rots = [rot3d_z(0.1), rot3d_x(0.2), rot3d_y(0.3)]
        result = filter_orientations(rots)
        assert len(result) == 3
        assert all(is_SO3(r) for r in result)


# ================================================================
# Question 4.3 — get_planet_orienations
# ================================================================

class TestGetPlanetOrientations:
    def test_single_planet(self):
        """Single rotation relative to Earth = that rotation."""
        R = [rot3d_z(0.3)]
        result = get_planet_orienations(R)
        assert len(result) == 1
        assert np.allclose(result[0], R[0], atol=1e-6)

    def test_two_planets(self):
        """
        rots = [R1, R2] where R1 is planet1 relative to Earth,
        R2 is planet2 relative to planet1.
        Planet2 relative to Earth = R1 @ R2.
        """
        R1 = rot3d_z(0.3)
        R2 = rot3d_y(0.5)
        result = get_planet_orienations([R1, R2])
        assert len(result) == 2
        assert np.allclose(result[0], R1, atol=1e-6)
        assert np.allclose(result[1], R1 @ R2, atol=1e-6)

    def test_all_results_are_SO3(self):
        rots = [rot3d_z(0.1), rot3d_x(0.2), rot3d_y(0.3)]
        result = get_planet_orienations(rots)
        assert all(is_SO3(r) for r in result)

    def test_three_planets(self):
        R1, R2, R3 = rot3d_z(0.1), rot3d_y(0.2), rot3d_x(0.3)
        result = get_planet_orienations([R1, R2, R3])
        assert np.allclose(result[0], R1, atol=1e-6)
        assert np.allclose(result[1], R1 @ R2, atol=1e-6)
        assert np.allclose(result[2], R1 @ R2 @ R3, atol=1e-6)

    def test_identity_chain(self):
        rots = [np.eye(3), np.eye(3), np.eye(3)]
        result = get_planet_orienations(rots)
        assert all(np.allclose(r, np.eye(3), atol=1e-6) for r in result)


# ================================================================
# Run
# ================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])