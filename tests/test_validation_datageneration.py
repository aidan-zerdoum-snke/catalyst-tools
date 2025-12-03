# SnkeOS Internal
#
# Copyright (C) 2024-2025 Snke OS GmbH, Germany. All rights reserved.

"""Validation tests.

This file shall contain all validation tests for datageneration and be extended for specific projects
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
from clearml import Dataset

from tests.conftest import generate_data


@pytest.fixture(scope="session")
def generate_default_data():
    """Runs the default experiment for validation."""
    # If you have other datasets in your config, you should probably remove these here with e.g., "-datasets.dev"
    return generate_data(
        ["+datasets.train.max_number=1", "datasets.train.output_dataset.dataset_name=validation-train-dataset"]
    )


@pytest.mark.validation(
    "This test generates a clearML dataset with only one sliceset with the default data generation settings."
)
def test_default_datageneration(generate_default_data):
    """Runs the default datageneration."""
    _, _ = generate_default_data


@pytest.mark.validation(
    "This test downloads the created dataset from clearML, checks if the first data item has all relevant attributes "
    "and if the image value of the h5 file contains a numpy array."
)
def test_dataset_contains_h5(generate_default_data):
    """Checks if dataset contains h5 and has relevant data."""
    dataset_tasks, _ = generate_default_data
    dataset = Dataset.get(dataset_tasks[0])
    dataset_path = Path(dataset.get_local_copy())

    with h5py.File(dataset_path / dataset.list_files()[0], "r") as f:
        print(f"Keys: {f.keys()}")
        assert "image" in f
        assert "meta_patient" in f

        assert type(f["image"][()]) is np.ndarray
