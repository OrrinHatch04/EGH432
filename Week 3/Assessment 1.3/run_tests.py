#!/usr/bin/env python
"""
run_tests.py — Two-phase test runner for EGH432 Assessment 1.3.

Phase 1 — Gradescope mirror (gradescope_tests.py):
  Runs the 25 autograder-style tests and prints your score out of 10.
  Pause after so you can read the score before Phase 2 starts.

Phase 2 — Correctness & adaptability suite (test_questions.py):
  123 tests covering edge cases, analytical correctness, round-trips,
  monotonicity, tolerance boundaries, and per-function speed guards.
"""

import os
import sys
import time

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Gradescope scoring plugin
# ─────────────────────────────────────────────────────────────────────────────

# Map from scored test-function name → (question label, points)
_SCORED = {
    "test_q11_3_generated": ("Q1.1  is_twist",                  0.5),
    "test_q12_3_generated": ("Q1.2  SE3_to_twist",              0.5),
    "test_q13_3_generated": ("Q1.3  twist_to_SE3",              0.5),
    "test_q21_3_generated": ("Q2.1  SE3_twist_traj",            1.0),
    "test_q22_3_generated": ("Q2.2  SE3_traj_relative",         1.5),
    "test_q31_3_generated": ("Q3.1  quintic_properties_symbolic", 2.0),
    "test_q41_3_generated": ("Q4.1  rocket_diagnostics",        2.0),
    "test_q42_3_generated": ("Q4.2  rocket_pose",               2.0),
}


class _GradeScopePlugin:
    """Pytest plugin that tracks scored-test outcomes and prints a scoreboard."""

    def __init__(self):
        # label → (passed, points_possible)
        self._results: dict[str, tuple[bool, float]] = {}

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        # Extract the bare test function name from the node ID
        fn_name = report.nodeid.rsplit("::", 1)[-1]
        if fn_name in _SCORED:
            label, pts = _SCORED[fn_name]
            self._results[label] = (report.passed, pts)

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        if not self._results:
            return

        tw = terminalreporter
        tw.write_sep("=", "GRADESCOPE SCORE")

        total_earned = 0.0
        total_possible = 0.0
        col = 40

        tw.write_line(f"\n  {'Question':<{col}}  {'Earned':>6}  {'Max':>6}  Status")
        tw.write_line(f"  {'-'*col}  {'-'*6}  {'-'*6}  ------")

        for label, (passed, pts) in self._results.items():
            earned = pts if passed else 0.0
            total_earned += earned
            total_possible += pts
            status = "PASS" if passed else "FAIL"
            tw.write_line(f"  {label:<{col}}  {earned:>6.1f}  {pts:>6.1f}  {status}")

        # Fill in any scored tests that weren't collected (e.g. import error)
        for fn_name, (label, pts) in _SCORED.items():
            if label not in self._results:
                total_possible += pts
                tw.write_line(f"  {label:<{col}}  {'0.0':>6}  {pts:>6.1f}  NOT RUN")

        tw.write_line(f"\n  {'TOTAL':<{col}}  {total_earned:>6.1f}  {total_possible:>6.1f}")
        tw.write_sep("=", f"Score: {total_earned:.1f} / {total_possible:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

PAUSE_SECONDS = 5   # seconds to pause between phases so you can read the score

def _banner(text: str, width: int = 70):
    bar = "=" * width
    print(f"\n{bar}\n  {text}\n{bar}\n", flush=True)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    # ── Phase 1: Gradescope mirror ────────────────────────────────────────────
    _banner("PHASE 1 — Gradescope autograder mirror (10 pts total)")
    gs_plugin = _GradeScopePlugin()
    gs_code = pytest.main(
        [
            "gradescope_tests.py",
            "-v",
            "-s",               # allow print() for inline timing output
            "--tb=short",
            "--no-header",
        ],
        plugins=[gs_plugin],
    )

    _banner(f"Pausing {PAUSE_SECONDS} s — read your Gradescope score above ↑")
    for remaining in range(PAUSE_SECONDS, 0, -1):
        sys.stdout.write(f"\r  Continuing in {remaining} s ...   ")
        sys.stdout.flush()
        time.sleep(1)
    print("\r" + " " * 40)   # clear the countdown line

    # ── Phase 2: Correctness & adaptability suite ─────────────────────────────
    _banner("PHASE 2 — Correctness & adaptability suite")
    tq_code = pytest.main(
        [
            "test_questions.py",
            "-v",
            "-s",
            "--tb=short",
            "--durations=10",   # show the 10 slowest tests at the end
            "--no-header",
        ],
    )

    _banner("All done!")
    sys.exit(max(gs_code, tq_code))
