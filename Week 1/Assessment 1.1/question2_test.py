#!/usr/bin/env python
### Manual Test Suite for question2.py | Orrin Hatch ###

print("Starting...", flush=True)
import os, sys
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TEST_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

print("Importing numpy...", flush=True)
import numpy as np
print("Numpy OK:", np.__version__, flush=True)

from question2 import rotation_create, translation_create, TransformCreate, transform_composer, TransformComposer

passed = 0
failed = 0

def check(name, condition, expected=None, got=None):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        if expected is not None:
            print(f"        Expected : {expected}")
            print(f"        Got      : {got}")
        failed += 1

def is_rotation_matrix(R, tol=1e-9):
    """Valid SO(3): R^T R = I and det(R) = +1."""
    return (R.shape == (3, 3)
            and np.allclose(R.T @ R, np.eye(3), atol=tol)
            and np.isclose(np.linalg.det(R), 1.0, atol=tol))

def is_valid_se3(T, tol=1e-9):
    """Valid SE(3): valid rotation block and bottom row [0,0,0,1]."""
    return (T.shape == (4, 4)
            and is_rotation_matrix(T[:3, :3], tol)
            and np.allclose(T[3, :], [0, 0, 0, 1], atol=tol))


# ================================================================
# Q2.1 — rotation_create
# ================================================================
print("\n--- Q2.1  rotation_create ---")

for axis in ["x", "y", "z"]:
    R = rotation_create(0.5, axis)
    check(f"shape  axis={axis}",   R.shape == (3, 3),          "(3,3)",    R.shape)
    check(f"dtype  axis={axis}",   R.dtype == np.float64,      "float64",  R.dtype)

for axis in ["x", "y", "z"]:
    for angle in [0.0, 0.3, np.pi/4, np.pi/2, np.pi, -np.pi/3]:
        R = rotation_create(angle, axis)
        check(f"SO(3)  axis={axis} angle={angle:.2f}", is_rotation_matrix(R))

for axis in ["x", "y", "z"]:
    R = rotation_create(0.0, axis)
    check(f"identity at 0  axis={axis}", np.allclose(R, np.eye(3)))

Rx = rotation_create(np.pi / 2, "x")
check("known Rx 90°", np.allclose(Rx, [[1,0,0],[0,0,-1],[0,1,0]], atol=1e-9),
      [[1,0,0],[0,0,-1],[0,1,0]], np.round(Rx, 4).tolist())

Ry = rotation_create(np.pi / 2, "y")
check("known Ry 90°", np.allclose(Ry, [[0,0,1],[0,1,0],[-1,0,0]], atol=1e-9),
      [[0,0,1],[0,1,0],[-1,0,0]], np.round(Ry, 4).tolist())

Rz = rotation_create(np.pi / 2, "z")
check("known Rz 90°", np.allclose(Rz, [[0,-1,0],[1,0,0],[0,0,1]], atol=1e-9),
      [[0,-1,0],[1,0,0],[0,0,1]], np.round(Rz, 4).tolist())

for axis in ["x", "y", "z"]:
    R = rotation_create(0.7, axis)
    R_inv = rotation_create(-0.7, axis)
    check(f"R(-a)==R(a).T  axis={axis}", np.allclose(R_inv, R.T, atol=1e-9))

try:
    rotation_create(0.5, "w")
    check("invalid axis raises", False, "ValueError", "no error raised")
except (ValueError, KeyError, UnboundLocalError):
    check("invalid axis raises", True)


# ================================================================
# Q2.2 — translation_create
# ================================================================
print("\n--- Q2.2  translation_create ---")

for axis in ["x", "y", "z"]:
    t = translation_create(1.0, axis)
    check(f"shape  axis={axis}", t.shape == (3, 1), "(3,1)", t.shape)

t = translation_create(5.0, "x")
check("x value", np.allclose(t, [[5.0],[0.0],[0.0]]),   [[5],[0],[0]], t.T.tolist())

t = translation_create(3.0, "y")
check("y value", np.allclose(t, [[0.0],[3.0],[0.0]]),   [[0],[3],[0]], t.T.tolist())

t = translation_create(7.5, "z")
check("z value", np.allclose(t, [[0.0],[0.0],[7.5]]),   [[0],[0],[7.5]], t.T.tolist())

t = translation_create(0.0, "x")
check("zero translation", np.allclose(t, np.zeros((3,1))))

t = translation_create(-2.5, "x")
check("negative value", np.allclose(t, [[-2.5],[0.0],[0.0]]), [[-2.5],[0],[0]], t.T.tolist())


# ================================================================
# Q2.3 — TransformCreate
# ================================================================
print("\n--- Q2.3  TransformCreate ---")

for axis in ["x", "y", "z"]:
    T = TransformCreate("r", axis).evaluate(0.5)
    check(f"r shape  axis={axis}", T.shape == (4,4),         "(4,4)",   T.shape)
    check(f"r dtype  axis={axis}", T.dtype == np.float64,    "float64", T.dtype)
    check(f"r SE(3)  axis={axis}", is_valid_se3(T))

for axis in ["x", "y", "z"]:
    T = TransformCreate("t", axis).evaluate(2.0)
    check(f"t shape  axis={axis}", T.shape == (4,4),         "(4,4)",   T.shape)
    check(f"t SE(3)  axis={axis}", is_valid_se3(T))

for axis in ["x", "y", "z"]:
    angle = 0.6
    T = TransformCreate("r", axis).evaluate(angle)
    R = rotation_create(angle, axis)
    check(f"rotation block correct  axis={axis}", np.allclose(T[:3,:3], R, atol=1e-9))

for i, axis in enumerate(["x", "y", "z"]):
    T = TransformCreate("t", axis).evaluate(3.0)
    expected = np.zeros(3); expected[i] = 3.0
    check(f"translation block correct  axis={axis}",
          np.allclose(T[:3, 3], expected, atol=1e-9), expected, T[:3,3])

for axis in ["x", "y", "z"]:
    check(f"r identity at 0  axis={axis}",
          np.allclose(TransformCreate("r", axis).evaluate(0.0), np.eye(4)))
    check(f"t identity at 0  axis={axis}",
          np.allclose(TransformCreate("t", axis).evaluate(0.0), np.eye(4)))

for axis in ["x", "y", "z"]:
    T = TransformCreate("r", axis).evaluate(0.5)
    check(f"bottom row correct r axis={axis}",
          np.allclose(T[3,:], [0,0,0,1]))


# ================================================================
# Q2.4 — transform_composer
# ================================================================
print("\n--- Q2.4  transform_composer ---")

aTb = TransformCreate("r", "z")
bTc = TransformCreate("t", "x")

T = transform_composer(aTb, np.pi/2, bTc, 1.0)
check("shape",      T.shape == (4,4),       "(4,4)",   T.shape)
check("dtype",      T.dtype == np.float64,  "float64", T.dtype)
check("is SE(3)",   is_valid_se3(T))

T = transform_composer(TransformCreate("r","z"), 0.0, TransformCreate("t","x"), 0.0)
check("both zero -> identity", np.allclose(T, np.eye(4)))

# Verify equals manual multiplication
a1, a2 = 0.7, 2.5
T  = transform_composer(aTb, a1, bTc, a2)
T_manual = aTb.evaluate(a1) @ bTc.evaluate(a2)
check("equals manual multiply", np.allclose(T, T_manual, atol=1e-12))

T1 = transform_composer(aTb, np.pi/4, bTc, 1.0)
T2 = transform_composer(bTc, 1.0, aTb, np.pi/4)
check("non-commutative", not np.allclose(T1, T2))

for axis in ["x", "y", "z"]:
    T = transform_composer(TransformCreate("r", axis), 0.4,
                           TransformCreate("t", axis), 1.5)
    check(f"SE(3) valid  r+t axis={axis}", is_valid_se3(T))


# ================================================================
# Q2.5 — TransformComposer
# ================================================================
print("\n--- Q2.5  TransformComposer ---")

transforms = [
    TransformCreate("r", "z"),
    TransformCreate("t", "x"),
    TransformCreate("r", "z"),
    TransformCreate("t", "x"),
]
composer = TransformComposer(transforms)
args     = [np.pi/4, 1.0, np.pi/3, 0.5]

T = composer.evaluate(args)
check("shape",      T.shape == (4,4),       "(4,4)",   T.shape)
check("dtype",      T.dtype == np.float64,  "float64", T.dtype)
check("is SE(3)",   is_valid_se3(T))

check("zeros -> identity", np.allclose(composer.evaluate([0.0,0.0,0.0,0.0]), np.eye(4)))

# Verify matches manual sequential chain
T_manual = np.eye(4)
for t, a in zip(transforms, args):
    T_manual = T_manual @ t.evaluate(a)
check("matches manual chain", np.allclose(T, T_manual, atol=1e-12))

# Single transform edge case
single = TransformComposer([TransformCreate("r", "z")])
check("single transform",
      np.allclose(single.evaluate([np.pi/2]),
                  TransformCreate("r","z").evaluate(np.pi/2)))

# Bottom row must always be [0,0,0,1] for random inputs
np.random.seed(42)
all_ok = True
for _ in range(20):
    rand_args = np.random.uniform(-np.pi, np.pi, 4).tolist()
    if not np.allclose(composer.evaluate(rand_args)[3,:], [0,0,0,1], atol=1e-9):
        all_ok = False
        break
check("bottom row [0,0,0,1] for 20 random inputs", all_ok)

# SE(3) validity for random inputs
all_se3 = True
np.random.seed(0)
for _ in range(20):
    rand_args = np.random.uniform(-np.pi, np.pi, 4).tolist()
    if not is_valid_se3(composer.evaluate(rand_args)):
        all_se3 = False
        break
check("SE(3) valid for 20 random inputs", all_se3)


# ================================================================
# Results
# ================================================================
total = passed + failed
print(f"\n{'='*50}")
print(f"  {passed}/{total} tests passed", "✓" if failed == 0 else "✗")
if failed > 0:
    print(f"  {failed} test(s) failed — review above")
print(f"{'='*50}\n")