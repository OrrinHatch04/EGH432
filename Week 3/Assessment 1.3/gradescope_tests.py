"""
gradescope_tests.py — Local mirror of the Gradescope autograder.

Replicates the exact 25-item structure shown in the autograder results:

  1.  Check submitted files                     (0 pts)
  2.  Q1.1 Test 1) is_twist import              (0 pts)
  3.  Q1.1 Test 2) is_twist sample              (0 pts)
  4.  Q1.1 Test 3) is_twist generated data      (0.5 pts)
  5.  Q1.2 Test 1) SE3_to_twist import          (0 pts)
  6.  Q1.2 Test 2) SE3_to_twist sample          (0 pts)
  7.  Q1.2 Test 3) SE3_to_twist generated data  (0.5 pts)
  8.  Q1.3 Test 1) twist_to_SE3 import          (0 pts)
  9.  Q1.3 Test 2) twist_to_SE3 sample          (0 pts)
  10. Q1.3 Test 3) twist_to_SE3 generated data  (0.5 pts)
  11. Q2.1 Test 1) SE3_twist_traj import        (0 pts)
  12. Q2.1 Test 2) SE3_twist_traj sample        (0 pts)
  13. Q2.1 Test 3) SE3_twist_traj generated     (1.0 pts)
  14. Q2.2 Test 1) SE3_traj_relative import     (0 pts)
  15. Q2.2 Test 2) SE3_traj_relative sample     (0 pts)
  16. Q2.2 Test 3) SE3_traj_relative generated  (1.5 pts)
  17. Q3.1 Test 1) quintic_properties import    (0 pts)
  18. Q3.1 Test 2) quintic_properties sample    (0 pts)
  19. Q3.1 Test 3) quintic_properties generated (2.0 pts)
  20. Q4.1 Test 1) rocket_diagnostics import    (0 pts)
  21. Q4.1 Test 2) rocket_diagnostics sample    (0 pts)
  22. Q4.1 Test 3) rocket_diagnostics generated (2.0 pts)
  23. Q4.2 Test 1) rocket_pose import           (0 pts)
  24. Q4.2 Test 2) rocket_pose sample           (0 pts)
  25. Q4.2 Test 3) rocket_pose generated        (2.0 pts)
  ──────────────────────────────────────────────────────
  Total                                         10.0 pts

Each Test 3 prints the minimum wall-clock time over N_TIMING repetitions.
"""

import os
import time

import numpy as np
import pytest
import spatialmath as sm
import spatialmath.base as smb

# Number of repetitions used to estimate minimum computation time in Test 3s
N_TIMING = 20


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _min_time(fn, n: int = N_TIMING) -> float:
    """Return the minimum wall-clock time (seconds) over *n* calls to fn()."""
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _make_se3(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0) -> sm.SE3:
    return sm.SE3.Trans(tx, ty, tz) * sm.SE3.Rz(rz) * sm.SE3.Ry(ry) * sm.SE3.Rx(rx)


def _build_rocket_traj(world_traj: sm.SE3) -> sm.SE3:
    """Build the relative-pose rocket traj: [world[0], rel[0], ..., rel[n-1]]."""
    from questions import SE3_traj_relative
    rel = SE3_traj_relative(world_traj)
    traj = sm.SE3.Empty()
    traj.append(world_traj[0])
    for r in rel:
        traj.append(r)
    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Item 1 — Check submitted files
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSubmittedFiles:
    """1. Check submitted files (0/0) — All required files submitted!"""

    def test_questions_py_present(self):
        here = os.path.dirname(os.path.abspath(__file__))
        assert os.path.isfile(os.path.join(here, "questions.py")), \
            "questions.py not found in submission directory"


# ─────────────────────────────────────────────────────────────────────────────
# Q1.1 — is_twist
# ─────────────────────────────────────────────────────────────────────────────

class TestQ11IstwistImport:
    """2. Q1.1 Test 1) Test that the is_twist method can be imported (0/0)"""

    def test_q11_1_import(self):
        from questions import is_twist
        assert callable(is_twist)


class TestQ11IstwistSample:
    """3. Q1.1 Test 2) Test that the is_twist method works for the sample (0/0)"""

    def test_q11_2_sample(self):
        from questions import is_twist
        # Sample: a valid 6-element numpy array is a twist
        assert is_twist(np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])) is True
        # Sample: a Python list of 6 numbers is NOT a twist
        assert is_twist([1, 2, 3, 4, 5, 6]) is False
        # Sample: a numpy array of wrong length is NOT a twist
        assert is_twist(np.zeros(3)) is False


class TestQ11IstwistGenerated:
    """4. Q1.1 Test 3) Test the is_twist method on generated data (0.5/0.5)"""

    def test_q11_3_generated(self):
        from questions import is_twist

        rng = np.random.default_rng(42)

        # Valid: shape (6,) numpy arrays of any dtype
        for _ in range(30):
            v = rng.standard_normal(6)
            assert is_twist(v) is True
        assert is_twist(np.array([0, 0, 0, 0, 0, 1], dtype=int)) is True
        assert is_twist(np.zeros(6, dtype=np.float32)) is True

        # Invalid: wrong shapes
        for shape in [(3,), (5,), (7,), (0,), (6, 1), (1, 6), (2, 3)]:
            assert is_twist(np.zeros(shape)) is False, f"Should be False for shape {shape}"

        # Invalid: wrong types
        for bad in [[1, 2, 3, 4, 5, 6], (1, 2, 3, 4, 5, 6), None, 6, 1.0, "twist"]:
            assert is_twist(bad) is False, f"Should be False for {type(bad)}"

        # Minimum computation time
        v = np.ones(6)
        t_min = _min_time(lambda: is_twist(v))
        print(f"\n    ⏱  is_twist min time ({N_TIMING} runs): {t_min*1e6:.3f} µs")


# ─────────────────────────────────────────────────────────────────────────────
# Q1.2 — SE3_to_twist
# ─────────────────────────────────────────────────────────────────────────────

class TestQ12SE3ToTwistImport:
    """5. Q1.2 Test 1) Test that the SE3_to_twist method can be imported (0/0)"""

    def test_q12_1_import(self):
        from questions import SE3_to_twist
        assert callable(SE3_to_twist)


class TestQ12SE3ToTwistSample:
    """6. Q1.2 Test 2) Test that the SE3_to_twist method works for the sample (0/0)"""

    def test_q12_2_sample(self):
        from questions import SE3_to_twist
        # Sample: identity matrix → zero twist
        twist = SE3_to_twist(np.eye(4))
        np.testing.assert_allclose(twist, np.zeros(6), atol=1e-12)
        # Sample: pure x-translation of 1 → [1,0,0, 0,0,0]
        twist = SE3_to_twist(sm.SE3.Tx(1.0).A)
        np.testing.assert_allclose(twist, [1, 0, 0, 0, 0, 0], atol=1e-12)


class TestQ12SE3ToTwistGenerated:
    """7. Q1.2 Test 3) Test the SE3_to_twist method on generated data (0.5/0.5)"""

    def test_q12_3_generated(self):
        from questions import SE3_to_twist, twist_to_SE3

        rng = np.random.default_rng(7)

        # Output must be a 1-D numpy array of length 6
        twist = SE3_to_twist(np.eye(4))
        assert isinstance(twist, np.ndarray) and twist.shape == (6,)

        # Pure translations: angular part must be zero
        for axis in ["x", "y", "z"]:
            d = rng.uniform(0.1, 3.0)
            T = getattr(sm.SE3, f"T{axis}")(d).A
            tw = SE3_to_twist(T)
            assert tw.shape == (6,)
            np.testing.assert_allclose(tw[3:], np.zeros(3), atol=1e-10,
                                       err_msg=f"Angular part should be 0 for pure T{axis}")

        # Pure rotations: angular magnitude must equal rotation angle
        for axis in ["x", "y", "z"]:
            angle = rng.uniform(0.05, 1.5)
            T = getattr(sm.SE3, f"R{axis}")(angle).A
            tw = SE3_to_twist(T)
            np.testing.assert_allclose(np.linalg.norm(tw[3:]), angle, atol=1e-10)

        # Round-trip: SE3 → twist → SE3 must recover the original
        for _ in range(10):
            t_vec = rng.uniform(-2, 2, 3)
            angles = rng.uniform(-0.8, 0.8, 3)
            T_orig = _make_se3(*t_vec, *angles).A
            tw = SE3_to_twist(T_orig)
            T_back = twist_to_SE3(tw)
            np.testing.assert_allclose(T_back, T_orig, atol=1e-9)

        # Minimum computation time
        T_bench = (sm.SE3.Trans(1, 2, 3) * sm.SE3.Rz(0.5)).A
        t_min = _min_time(lambda: SE3_to_twist(T_bench))
        print(f"\n    ⏱  SE3_to_twist min time ({N_TIMING} runs): {t_min*1000:.4f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Q1.3 — twist_to_SE3
# ─────────────────────────────────────────────────────────────────────────────

class TestQ13TwistToSE3Import:
    """8. Q1.3 Test 1) Test that the twist_to_SE3 method can be imported (0/0)"""

    def test_q13_1_import(self):
        from questions import twist_to_SE3
        assert callable(twist_to_SE3)


class TestQ13TwistToSE3Sample:
    """9. Q1.3 Test 2) Test that the twist_to_SE3 method works for the sample (0/0)"""

    def test_q13_2_sample(self):
        from questions import twist_to_SE3
        # Sample: zero twist → identity
        np.testing.assert_allclose(twist_to_SE3(np.zeros(6)), np.eye(4), atol=1e-12)
        # Sample: pure x-translation → Tx(2)
        T = twist_to_SE3(np.array([2.0, 0, 0, 0, 0, 0]))
        np.testing.assert_allclose(T, sm.SE3.Tx(2.0).A, atol=1e-12)


class TestQ13TwistToSE3Generated:
    """10. Q1.3 Test 3) Test the twist_to_SE3 method on generated data (0.5/0.5)"""

    def test_q13_3_generated(self):
        from questions import twist_to_SE3, SE3_to_twist

        rng = np.random.default_rng(13)

        # Output must be 4×4
        assert twist_to_SE3(np.zeros(6)).shape == (4, 4)

        # Pure rotations
        for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
            angle = rng.uniform(0.05, 1.5)
            tw = np.zeros(6); tw[3 + idx] = angle
            T = twist_to_SE3(tw)
            np.testing.assert_allclose(T, getattr(sm.SE3, f"R{axis}")(angle).A, atol=1e-12)

        # Output must be a valid SE3 matrix (rotation orthogonal, det=1, bottom row [0,0,0,1])
        for _ in range(20):
            tw = rng.uniform(-1, 1, 6)
            T = twist_to_SE3(tw)
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)
            np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)

        # Round-trip: twist → SE3 → twist must recover the original
        for _ in range(10):
            tw_orig = rng.uniform(-1, 1, 6)
            T = twist_to_SE3(tw_orig)
            tw_back = SE3_to_twist(T)
            np.testing.assert_allclose(tw_back, tw_orig, atol=1e-9)

        # Minimum computation time
        tw_bench = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        t_min = _min_time(lambda: twist_to_SE3(tw_bench))
        print(f"\n    ⏱  twist_to_SE3 min time ({N_TIMING} runs): {t_min*1000:.4f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Q2.1 — SE3_twist_traj
# ─────────────────────────────────────────────────────────────────────────────

class TestQ21SE3TwistTrajImport:
    """11. Q2.1 Test 1) Test that the SE3_twist_traj method can be imported (0/0)"""

    def test_q21_1_import(self):
        from questions import SE3_twist_traj
        assert callable(SE3_twist_traj)


class TestQ21SE3TwistTrajSample:
    """12. Q2.1 Test 2) Test that the SE3_twist_traj method works for the sample (0/0)"""

    def test_q21_2_sample(self):
        from questions import SE3_twist_traj
        wTa = sm.SE3()
        wTb = sm.SE3.Tx(1.0)
        traj = SE3_twist_traj(wTa, wTb, 5)
        # Must return an sm.SE3 with 5 poses
        assert isinstance(traj, sm.SE3)
        assert len(traj) == 5
        # First pose = wTa, last pose = wTb
        np.testing.assert_allclose(traj[0].A, wTa.A, atol=1e-10)
        np.testing.assert_allclose(traj[-1].A, wTb.A, atol=1e-10)


class TestQ21SE3TwistTrajGenerated:
    """13. Q2.1 Test 3) Test the SE3_twist_traj method on generated data (1/1)"""

    def test_q21_3_generated(self):
        from questions import SE3_twist_traj

        rng = np.random.default_rng(21)

        for trial in range(8):
            n = rng.integers(5, 25)
            t_vec = rng.uniform(-3, 3, 3)
            angles = rng.uniform(-0.6, 0.6, 3)
            wTa = _make_se3(*rng.uniform(-1, 1, 3), *rng.uniform(-0.3, 0.3, 3))
            wTb = _make_se3(*t_vec, *angles)

            traj = SE3_twist_traj(wTa, wTb, int(n))

            assert isinstance(traj, sm.SE3), "Must return sm.SE3"
            assert len(traj) == n, f"Length must be n={n}"
            np.testing.assert_allclose(traj[0].A, wTa.A, atol=1e-9,
                                       err_msg=f"Trial {trial}: first pose must equal wTa")
            np.testing.assert_allclose(traj[-1].A, wTb.A, atol=1e-9,
                                       err_msg=f"Trial {trial}: last pose must equal wTb")

            # All intermediate poses must be valid SE3 matrices
            for i in range(len(traj)):
                R = traj[i].A[:3, :3]
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)
                np.testing.assert_allclose(traj[i].A[3], [0, 0, 0, 1], atol=1e-10)

        # Minimum computation time (n=21 — same as autograder fixture)
        wTa = sm.SE3()
        wTb = _make_se3(6.0, 3.0, 1.0, 0.0, 0.0, np.pi / 3)
        t_min = _min_time(lambda: SE3_twist_traj(wTa, wTb, 21))
        print(f"\n    ⏱  SE3_twist_traj (n=21) min time ({N_TIMING} runs): {t_min*1000:.4f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Q2.2 — SE3_traj_relative
# ─────────────────────────────────────────────────────────────────────────────

class TestQ22SE3TrajRelativeImport:
    """14. Q2.2 Test 1) Test that the SE3_traj_relative method can be imported (0/0)"""

    def test_q22_1_import(self):
        from questions import SE3_traj_relative
        assert callable(SE3_traj_relative)


class TestQ22SE3TrajRelativeSample:
    """15. Q2.2 Test 2) Test that the SE3_traj_relative method works for the sample (0/0)"""

    def test_q22_2_sample(self):
        from questions import SE3_traj_relative
        # Sample: 3-pose straight-line trajectory
        traj = sm.SE3.Empty()
        for k in range(3):
            traj.append(sm.SE3.Tx(float(k)))
        rel = SE3_traj_relative(traj)
        assert isinstance(rel, sm.SE3)
        assert len(rel) == 2
        # Each relative step should be Tx(1)
        for i in range(2):
            np.testing.assert_allclose(rel[i].A, sm.SE3.Tx(1.0).A, atol=1e-10)


class TestQ22SE3TrajRelativeGenerated:
    """16. Q2.2 Test 3) Test the SE3_traj_relative method on generated data (1.5/1.5)"""

    def test_q22_3_generated(self):
        from questions import SE3_traj_relative, SE3_twist_traj

        rng = np.random.default_rng(22)

        for trial in range(8):
            n = rng.integers(5, 20)
            wTa = _make_se3(*rng.uniform(-1, 1, 3), *rng.uniform(-0.3, 0.3, 3))
            wTb = _make_se3(*rng.uniform(-3, 3, 3), *rng.uniform(-0.6, 0.6, 3))
            world = SE3_twist_traj(wTa, wTb, int(n))

            rel = SE3_traj_relative(world)

            assert isinstance(rel, sm.SE3), "Must return sm.SE3"
            assert len(rel) == len(world) - 1, "Length must be n-1"

            # Reconstruction: composing relative poses must recover all world poses
            current = world[0]
            for i in range(len(rel)):
                current = current * rel[i]
                np.testing.assert_allclose(
                    current.A, world[i + 1].A, atol=1e-7,
                    err_msg=f"Trial {trial}: reconstruction mismatch at index {i+1}",
                )

        # Minimum computation time (n=21 trajectory)
        wTa = sm.SE3()
        wTb = _make_se3(6.0, 3.0, 1.0, 0.0, 0.0, np.pi / 3)
        world = SE3_twist_traj(wTa, wTb, 21)
        t_min = _min_time(lambda: SE3_traj_relative(world))
        print(f"\n    ⏱  SE3_traj_relative (n=21) min time ({N_TIMING} runs): {t_min*1000:.4f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Q3.1 — quintic_properties_symbolic
# ─────────────────────────────────────────────────────────────────────────────

class TestQ31QuinticImport:
    """17. Q3.1 Test 1) Test that the quintic_properties_symbolic method can be imported (0/0)"""

    def test_q31_1_import(self):
        from questions import quintic_properties_symbolic
        assert callable(quintic_properties_symbolic)


class TestQ31QuinticSample:
    """18. Q3.1 Test 2) Test that the quintic_properties_symbolic method works for the sample (0/0)"""

    def test_q31_2_sample(self):
        from questions import quintic_properties_symbolic
        # Sample: standard unit-step quintic q(t) = 6t^5 - 15t^4 + 10t^3
        # Analytically: max|q''| = 10√3/3 ≈ 5.7735, max|q'''| = 60
        acc, jerk = quintic_properties_symbolic((6, -15, 10, 0, 0, 0))
        assert len((acc, jerk)) == 2
        np.testing.assert_allclose(acc, 10.0 * np.sqrt(3) / 3, rtol=1e-5)
        np.testing.assert_allclose(jerk, 60.0, rtol=1e-5)


class TestQ31QuinticGenerated:
    """19. Q3.1 Test 3) Test the quintic_properties_symbolic method on generated data (2/2)"""

    def test_q31_3_generated(self):
        from questions import quintic_properties_symbolic

        rng = np.random.default_rng(31)

        # Return type checks
        result = quintic_properties_symbolic((6, -15, 10, 0, 0, 0))
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

        # Known analytical results
        # q(t) = t^2 → q'' = 2 (constant), q''' = 0
        acc, jerk = quintic_properties_symbolic((0, 0, 0, 1, 0, 0))
        np.testing.assert_allclose(acc, 2.0, rtol=1e-5)
        np.testing.assert_allclose(jerk, 0.0, atol=1e-8)

        # q(t) = t^3 → q'' = 6t (max=6 at t=1), q''' = 6
        acc, jerk = quintic_properties_symbolic((0, 0, 1, 0, 0, 0))
        np.testing.assert_allclose(acc, 6.0, rtol=1e-5)
        np.testing.assert_allclose(jerk, 6.0, rtol=1e-5)

        # Zero polynomial → both zero
        acc, jerk = quintic_properties_symbolic((0, 0, 0, 0, 0, 0))
        assert acc == pytest.approx(0.0)
        assert jerk == pytest.approx(0.0)

        # Results must always be non-negative (they are max absolute values)
        for _ in range(10):
            coeffs = tuple(rng.uniform(-5, 5, 6))
            acc, jerk = quintic_properties_symbolic(coeffs)
            assert acc >= 0.0, f"Acceleration must be non-negative, got {acc}"
            assert jerk >= 0.0, f"Jerk must be non-negative, got {jerk}"

        # Negating coefficients must not change the magnitudes
        std = (6.0, -15.0, 10.0, 0.0, 0.0, 0.0)
        neg = tuple(-c for c in std)
        a1, j1 = quintic_properties_symbolic(std)
        a2, j2 = quintic_properties_symbolic(neg)
        np.testing.assert_allclose(a1, a2, rtol=1e-5)
        np.testing.assert_allclose(j1, j2, rtol=1e-5)

        # Minimum computation time
        t_min = _min_time(lambda: quintic_properties_symbolic((6, -15, 10, 0, 0, 0)))
        print(f"\n    ⏱  quintic_properties_symbolic min time ({N_TIMING} runs): {t_min*1000:.4f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Q4.1 — rocket_diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class TestQ41RocketDiagnosticsImport:
    """20. Q4.1 Test 1) Test that the rocket_diagnostics method can be imported (0/0)"""

    def test_q41_1_import(self):
        from questions import rocket_diagnostics
        assert callable(rocket_diagnostics)


class TestQ41RocketDiagnosticsSample:
    """21. Q4.1 Test 2) Test that the rocket_diagnostics method works for the sample (0/0)"""

    def test_q41_2_sample(self):
        from questions import rocket_diagnostics
        # Sample: [I, Tx(1), Tx(1)] → cumprods: Tx(1) at idx=1 (t=0.5), Tx(2) at idx=2 (t=1.0)
        # But traj[0]=I (the starting world pose) gives cumprod=I at idx=0 (t=0.0)
        step = sm.SE3.Tx(1.0)
        traj = sm.SE3.Empty()
        traj.append(sm.SE3())     # world pose at t=0 (prepended start)
        traj.append(step)          # relative step 1
        traj.append(step)          # relative step 2
        # Target: identity → found at index 0 → t = 0.0 s
        assert rocket_diagnostics(traj, sm.SE3()) == pytest.approx(0.0)
        # Target: Tx(1) → found at index 1 → t = 0.5 s
        assert rocket_diagnostics(traj, sm.SE3.Tx(1.0)) == pytest.approx(0.5)
        # Target: Tx(2) → found at index 2 → t = 1.0 s
        assert rocket_diagnostics(traj, sm.SE3.Tx(2.0)) == pytest.approx(1.0)


class TestQ41RocketDiagnosticsGenerated:
    """22. Q4.1 Test 3) Test the rocket_diagnostics method on generated data (2/2)"""

    def test_q41_3_generated(self):
        from questions import rocket_diagnostics, SE3_twist_traj, SE3_traj_relative

        rng = np.random.default_rng(41)

        for trial in range(6):
            n = rng.integers(8, 20)
            wTa = _make_se3(*rng.uniform(-2, 2, 3), *rng.uniform(-0.4, 0.4, 3))
            wTb = _make_se3(*rng.uniform(-3, 3, 3), *rng.uniform(-0.5, 0.5, 3))
            world = SE3_twist_traj(wTa, wTb, int(n))
            traj = _build_rocket_traj(world)

            # Every world pose must be found at its expected time
            for idx in range(len(world)):
                expected_t = idx * 0.5
                t = rocket_diagnostics(traj, world[idx])
                assert t == pytest.approx(expected_t, abs=1e-6), \
                    f"Trial {trial}: world[{idx}] should be at t={expected_t}, got {t}"

            # A clearly absent pose must return None
            absent = _make_se3(999.0, 999.0, 999.0)
            assert rocket_diagnostics(traj, absent) is None

        # Minimum computation time (n=11 trajectory, searching for last pose)
        world = SE3_twist_traj(sm.SE3(), _make_se3(4, 0, 2, 0, 0, np.pi / 4), 11)
        traj = _build_rocket_traj(world)
        T_goal = world[-1]
        t_min = _min_time(lambda: rocket_diagnostics(traj, T_goal))
        print(f"\n    ⏱  rocket_diagnostics (n=11) min time ({N_TIMING} runs): {t_min*1000:.4f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Q4.2 — rocket_pose
# ─────────────────────────────────────────────────────────────────────────────

class TestQ42RocketPoseImport:
    """23. Q4.2 Test 1) Test that the rocket_diagnostics method can be imported (0/0)"""

    def test_q42_1_import(self):
        from questions import rocket_pose
        assert callable(rocket_pose)


class TestQ42RocketPoseSample:
    """24. Q4.2 Test 2) Test that the rocket_diagnostics method works for the sample (0/0)"""

    def test_q42_2_sample(self):
        from questions import rocket_pose
        # Sample: straight-line world trajectory [Tx(0), Tx(1), Tx(2), Tx(3), Tx(4)]
        traj = sm.SE3.Empty()
        for k in range(5):
            traj.append(sm.SE3.Tx(float(k)))
        # From Tx(2): 2 steps remaining → t = 1.0 s; aTg = Tx(-2)*Tx(4) = Tx(2)
        wTa = sm.SE3.Tx(2.0)
        aTg, t = rocket_pose(traj, wTa)
        assert t == pytest.approx(1.0)
        np.testing.assert_allclose(aTg.A, sm.SE3.Tx(2.0).A, atol=1e-8)
        # From last pose: 0 remaining → aTg = identity
        aTg_last, t_last = rocket_pose(traj, traj[-1])
        assert t_last == pytest.approx(0.0)
        np.testing.assert_allclose(aTg_last.A, np.eye(4), atol=1e-8)


class TestQ42RocketPoseGenerated:
    """25. Q4.2 Test 3) Test the rocket_pose method on generated data (2/2)"""

    def test_q42_3_generated(self):
        from questions import rocket_pose, SE3_twist_traj

        rng = np.random.default_rng(42)

        for trial in range(6):
            n = rng.integers(8, 22)
            wTa_s = _make_se3(*rng.uniform(-2, 2, 3), *rng.uniform(-0.4, 0.4, 3))
            wTb_e = _make_se3(*rng.uniform(-3, 3, 3), *rng.uniform(-0.5, 0.5, 3))
            traj = SE3_twist_traj(wTa_s, wTb_e, int(n))

            for idx in range(len(traj)):
                wTa = traj[idx]
                aTg, t = rocket_pose(traj, wTa)

                # Remaining time: (n-1-idx) × 0.5 s
                expected_t = (len(traj) - 1 - idx) * 0.5
                assert t == pytest.approx(expected_t, abs=1e-6), \
                    f"Trial {trial}, idx {idx}: expected t={expected_t}, got {t}"

                # aTg must be a valid SE3
                assert isinstance(aTg, sm.SE3)
                R = aTg.A[:3, :3]
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)

                # wTa · aTg must equal the goal (traj[-1])
                wTg_computed = wTa * aTg
                np.testing.assert_allclose(wTg_computed.A, traj[-1].A, atol=1e-7,
                                           err_msg=f"Trial {trial}, idx {idx}: wTa*aTg != wTg")

        # Minimum computation time
        traj = SE3_twist_traj(sm.SE3(), _make_se3(4, 0, 2, 0, 0, np.pi / 4), 21)
        wTa_mid = traj[10]
        t_min = _min_time(lambda: rocket_pose(traj, wTa_mid))
        print(f"\n    ⏱  rocket_pose (n=21, mid-traj) min time ({N_TIMING} runs): {t_min*1000:.4f} ms")
