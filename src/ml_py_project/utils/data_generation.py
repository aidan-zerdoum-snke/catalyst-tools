# SnkeOS Internal
#
# Copyright (C) 2024-2025 Snke OS GmbH, Germany. All rights reserved.

"""Datageneration."""

import hydra
from datageneration.datageneration import generate_clearml_datasets
from omegaconf import DictConfig


@hydra.main(config_path="../config/data", config_name="data_generation")
def main(cfg: DictConfig):
    """Generation of clearML dataset from patientlists."""
    generate_clearml_datasets(cfg)


if __name__ == "__main__":
    main()
