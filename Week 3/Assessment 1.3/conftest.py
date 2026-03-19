"""
conftest.py — pytest plugin for EGH432 Assessment 1.3 timing reports.

Hooks into every test call to measure actual wall-clock time, then prints:
  • A per-test timing table (sorted by duration, descending) at the end.
  • Inline timing annotations on the speed-specific tests via a fixture.
"""

import time
from collections import defaultdict

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Per-test wall-clock timing (captures every test, not just speed tests)
# ─────────────────────────────────────────────────────────────────────────────

_test_times: dict[str, float] = {}
_section_map = {
    "TestIsTwist":                   "Q1.1",
    "TestSE3ToTwist":                "Q1.2",
    "TestTwistToSE3":                "Q1.3",
    "TestSE3TwistTraj":              "Q2.1",
    "TestSE3TrajRelative":           "Q2.2",
    "TestQuinticPropertiesSymbolic": "Q3.1",
    "TestRocketDiagnostics":         "Q4.1",
    "TestRocketPose":                "Q4.2",
}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Wrap each test call to measure wall-clock duration."""
    t0 = time.perf_counter()
    yield
    _test_times[item.nodeid] = time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: minimum-time measurement for speed tests
# ─────────────────────────────────────────────────────────────────────────────

class _Timer:
    """Context-manager / callable helper used by the `timer` fixture."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed: float = float("inf")
        self._t0: float = 0.0

    # ── context-manager usage ──────────────────────────────────────────────
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0
        self._report()

    # ── min-of-N usage ─────────────────────────────────────────────────────
    def min_of(self, fn, n: int = 5):
        """Run *fn* N times and record the minimum wall-clock duration."""
        best = float("inf")
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        self.elapsed = best
        self._report()
        return best

    def _report(self):
        ms = self.elapsed * 1_000
        print(f"\n    ⏱  {self.label}: {ms:.3f} ms", flush=True)


_all_timers: list[tuple[str, _Timer]] = []


@pytest.fixture
def timer(request):
    """
    Inject a _Timer into any test that requests it.

    Usage — context manager::

        def test_speed_foo(timer):
            with timer("SE3_to_twist × 1 000"):
                for _ in range(1_000):
                    SE3_to_twist(T)
            assert timer.elapsed < 0.5

    Usage — min-of-N::

        def test_speed_bar(timer):
            timer.min_of(lambda: SE3_to_twist(T), n=5)
            assert timer.elapsed < 0.5
    """
    label = request.node.name
    t = _Timer(label)
    _all_timers.append((request.node.nodeid, t))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Terminal summary
# ─────────────────────────────────────────────────────────────────────────────

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a sorted timing table after the normal pytest summary."""
    if not _test_times:
        return

    # Group by question section
    sections: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for nodeid, elapsed in _test_times.items():
        section = "Other"
        for cls_name, tag in _section_map.items():
            if cls_name in nodeid:
                section = tag
                break
        # Short name: everything after the last "::"
        short = nodeid.rsplit("::", 1)[-1]
        sections[section].append((short, elapsed))

    tw = terminalreporter
    tw.write_sep("=", "Computation-time summary (wall-clock, sorted slowest→fastest)")

    col_w = 52
    for section in ["Q1.1", "Q1.2", "Q1.3", "Q2.1", "Q2.2", "Q3.1", "Q4.1", "Q4.2", "Other"]:
        tests = sections.get(section)
        if not tests:
            continue
        tw.write_line(f"\n  {section}")
        tw.write_line(f"  {'Test':<{col_w}}  {'Time (ms)':>10}  {'Time (s)':>10}")
        tw.write_line(f"  {'-'*col_w}  {'-'*10}  {'-'*10}")
        for name, elapsed in sorted(tests, key=lambda x: -x[1]):
            ms = elapsed * 1_000
            tw.write_line(f"  {name:<{col_w}}  {ms:>10.3f}  {elapsed:>10.6f}")

    total = sum(_test_times.values())
    tw.write_line(f"\n  {'TOTAL':<{col_w}}  {total*1000:>10.3f}  {total:>10.6f}")
    tw.write_sep("=", "end timing summary")
