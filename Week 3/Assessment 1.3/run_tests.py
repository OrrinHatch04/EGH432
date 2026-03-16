#!/usr/bin/env python
"""
run_tests.py
"""
import pytest
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pytest.main(["test_questions.py", "-v"])