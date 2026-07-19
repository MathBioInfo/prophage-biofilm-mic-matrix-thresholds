#!/usr/bin/env python3
"""Fast checks for the analytical threshold functions.

This script does not regenerate all figures. It only verifies a few numerical
values used in the manuscript and README.
"""

from __future__ import annotations

from generate_figures import threshold_rrho


def assert_close(name: str, actual: float, expected: float, tolerance: float = 1e-4) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{name}: got {actual:.8g}, expected {expected:.8g}")


def main() -> None:
    theta = 5.0
    assert_close("R_M=0.5, a=0.1", threshold_rrho(0.1, 0.5, theta), 2.0)
    assert_close("R_M=0.5, a=1", threshold_rrho(1.0, 0.5, theta), 1.4)
    assert_close("R_M=0.5, a=10", threshold_rrho(10.0, 0.5, theta), 1.057142857)
    assert_close("R_M=0.5, a=100", threshold_rrho(100.0, 0.5, theta), 1.005970149)
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
