"""Pytest fixtures for evaluation tests."""

import pytest


@pytest.fixture
def sample_predictions():
    return [
        ["Art. 1 ZGB", "BGE 116 Ia 56 E. 2b"],
        ["Art. 1 OR", "Art. 11 OR"],
        ["BGE 121 III 38 E. 2b"],
    ]


@pytest.fixture
def sample_gold():
    return [
        ["Art. 1 ZGB", "BGE 116 Ia 56 E. 2b"],    # Perfect match
        ["Art. 1 OR", "BGE 119 II 449 E. 3.4"],    # Partial match
        ["Art. 117 StGB", "BGE 121 III 38 E. 2b"], # Partial match
    ]