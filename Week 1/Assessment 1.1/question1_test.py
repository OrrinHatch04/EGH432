#!/usr/bin/env python
### Manual Test Suite for question1.py | Orrin Hatch ###

import os
import sys

# Always run from this file's directory so question1.py is importable
# regardless of where VS Code launches Python from
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TEST_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

CSV_FILE = os.path.join(TEST_DIR, "robots1.csv")

from question1 import list_to_dict, count_population, Robot, csv_to_robots, closest_robot

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


# ================================================================
# Q1.1 — list_to_dict
# ================================================================
print("\n--- Q1.1  list_to_dict ---")

result = list_to_dict(["hello", "hi"])
check("basic",          result == {"hello": 5, "hi": 2},       {"hello": 5, "hi": 2}, result)

result = list_to_dict([])
check("empty list",     result == {},                           {}, result)

result = list_to_dict(["a"])
check("single char",    result == {"a": 1},                    {"a": 1}, result)

result = list_to_dict([""])
check("empty string",   result == {"": 0},                     {"": 0}, result)

result = list_to_dict(["robot", "arm", "kinematics"])
check("values=lengths", all(result[w] == len(w) for w in result), "each value == len(key)", result)

result = list_to_dict(["hi", "hi", "hi"])
check("duplicates",     result == {"hi": 2},                   {"hi": 2}, result)

check("returns dict",   isinstance(list_to_dict(["test"]), dict))

words = [f"word{i}" for i in range(500)]
result = list_to_dict(words)
check("large list",     len(result) == 500 and all(result[w] == len(w) for w in words))


# ================================================================
# Q1.2 — count_population
# ================================================================
print("\n--- Q1.2  count_population ---")

pop = {"Australia": 26_000_000, "Japan": 125_000_000, "Germany": 84_000_000}

result = count_population(pop, ["Australia", "Japan"])
check("two countries",      result == 151_000_000,  151_000_000, result)

result = count_population(pop, ["Australia", "Canada"])
check("missing country",    result == 26_000_000,   26_000_000,  result)

result = count_population(pop, ["Canada", "France"])
check("all missing -> 0",   result == 0,            0,           result)

result = count_population(pop, [])
check("empty list -> 0",    result == 0,            0,           result)

result = count_population(pop, ["Germany"])
check("single country",     result == 84_000_000,   84_000_000,  result)

check("returns int",        isinstance(count_population(pop, ["Australia"]), int))

result = count_population({"China": 1_400_000_000, "India": 1_400_000_000}, ["China", "India"])
check("large populations",  result == 2_800_000_000, 2_800_000_000, result)


# ================================================================
# Q1.3 — Robot class
# ================================================================
print("\n--- Q1.3  Robot class ---")

r = Robot("Panda", 7, "revolute", 40000.0)
check("name stored",        r.name == "Panda",          "Panda",      r.name)
check("n stored",           r.n == 7,                   7,            r.n)
check("joint_type stored",  r.joint_type == "revolute", "revolute",   r.joint_type)
check("price stored",       r.price == 40000.0,         40000.0,      r.price)

expected_str = "Panda Robot Description: number of joints: 7, joint type: revolute"
check("__str__ Panda",      str(r) == expected_str,     expected_str, str(r))

r2 = Robot("Puma560", 6, "revolute", 19000.0)
expected_str2 = "Puma560 Robot Description: number of joints: 6, joint type: revolute"
check("__str__ Puma560",    str(r2) == expected_str2,   expected_str2, str(r2))

r3 = Robot("TestBot", 3, "prismatic", 5000.0)
check("prismatic in str",   "prismatic" in str(r3))

check("price as float",     Robot("X", 1, "mixed", 99.99).price == 99.99)


# ================================================================
# Q1.4 — csv_to_robots
# ================================================================
print("\n--- Q1.4  csv_to_robots ---")

robots = csv_to_robots(CSV_FILE)
rd = {r.name: r for r in robots}

check("count == 2",         len(robots) == 2,                   2,   len(robots))
check("returns Robot list", all(isinstance(r, Robot) for r in robots))
check("Panda exists",       "Panda" in rd)
check("Puma560 exists",     "Puma560" in rd)
check("Panda n == 7",       rd["Panda"].n == 7,                 7,   rd["Panda"].n)
check("Puma560 n == 6",     rd["Puma560"].n == 6,               6,   rd["Puma560"].n)
check("Panda price",        rd["Panda"].price == 40000.0,       40000.0, rd["Panda"].price)
check("Puma560 price",      rd["Puma560"].price == 19000.0,     19000.0, rd["Puma560"].price)
check("n is int",           all(isinstance(r.n, int) for r in robots))
check("price is float",     all(isinstance(r.price, float) for r in robots))
check("joint_type revolute",all(r.joint_type == "revolute" for r in robots))


# ================================================================
# Q1.5 — closest_robot
# ================================================================
print("\n--- Q1.5  closest_robot ---")

r = closest_robot(CSV_FILE, 25000)
check("limit=25000 -> Puma560",     r.name == "Puma560",  "Puma560", r.name)

r = closest_robot(CSV_FILE, 50000)
check("limit=50000 -> Panda",       r.name == "Panda",    "Panda",   r.name)

r = closest_robot(CSV_FILE, 40000)
check("limit=40000 exact -> Panda", r.name == "Panda",    "Panda",   r.name)

r = closest_robot(CSV_FILE, 19000)
check("limit=19000 exact -> Puma",  r.name == "Puma560",  "Puma560", r.name)

r = closest_robot(CSV_FILE, 5000)
check("limit=5000 below all",       r.name == "Puma560",  "Puma560", r.name)

check("returns Robot",              isinstance(closest_robot(CSV_FILE, 25000), Robot))


# ================================================================
# Results
# ================================================================
total = passed + failed
print(f"\n{'='*50}")
print(f"  {passed}/{total} tests passed", "✓" if failed == 0 else "✗")
if failed > 0:
    print(f"  {failed} test(s) failed — review above")
print(f"{'='*50}\n")