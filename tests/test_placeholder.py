# SnkeOS Internal
#
# Copyright (C) 2024-2025 Snke OS GmbH, Germany. All rights reserved.

"""Placeholder pytest file.

This file serves as a placeholder for your pytests, you might want to remove the placeholder tests once you have
implemented actual tests. Don't forget to mark your tests!
"""

import pytest
from ml_py_project.example import example_function


@pytest.mark.unit
@pytest.mark.parametrize(
    ("test_input", "expected"), [((3, 2), 1.5), ((2, 4), 0.5), pytest.param((1, 0), 42, marks=pytest.mark.xfail)]
)
def test_placeholder1(test_input, expected):
    """Parametrized (with expected fail case) unit test placeholder. Don't forget to mark your tests."""
    assert example_function(test_input[0], test_input[1]) == expected
