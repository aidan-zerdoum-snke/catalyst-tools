# SnkeOS Internal
#
# Copyright (C) 2024-2025 Snke OS GmbH, Germany. All rights reserved.

"""configuration functions for pytest."""

import os
import re
import shutil
import subprocess
import time

import pytest
from clearml import Task
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from snketorch.utils import utils


def pytest_collection_modifyitems(session, config, items):
    """Adds the validation descriptions to the user properties for report generation."""
    for item in items:
        for marker in item.iter_markers(name="validation"):
            description = marker.args[0]
            item.user_properties.append(("validation", description))
            item.user_properties.append(("subsection", item.parent.name.split("_")[-1].split(".")[0] + "_cases"))


utils.register_configs()


def run_experiment(overrides: list[str]) -> tuple[Task, DictConfig]:
    """Runs an experiment with given overrides."""
    # create local config (just for validation purposes)
    with initialize(config_path="../src/ml_py_project/config"):
        cfg = compose(config_name="config.yaml", return_hydra_config=True, overrides=overrides)

    result = subprocess.run(  # noqa: S603
        [  # noqa: RUF005
            str(shutil.which("python")),  # type: ignore[list-item]
            os.path.join(os.path.dirname(__file__), "../src/ml_py_project/main.py"),
        ]
        + overrides,
        capture_output=True,
        text=True,
    )
    print("Results: " + str(result))
    task_id = re.findall(r"task id=(\w+)", result.stdout)[0]

    remote_task = Task.get_task(task_id=task_id)

    wait_for_task_to_complete(remote_task)

    remote_task = Task.get_task(task_id=task_id)  # update information from server

    return remote_task, cfg


def wait_for_task_to_complete(remote_task: Task):
    """Waits for the clearML Task to be finished."""
    remote_conf = OmegaConf.create(remote_task.data.configuration["OmegaConf"].value)
    if remote_conf.mlops.queue_name is None or remote_conf.mlops.queue_name == "":
        return

    timeout = 600  # seconds
    wait_time = 1  # seconds
    counter = 0
    while (
        remote_task.get_status() != "completed"
        and remote_task.get_status() != "failed"
        and remote_task.get_status() != "stopped"
    ):
        time.sleep(wait_time)
        counter += wait_time
        if timeout < counter:
            raise TimeoutError("ClearML Task is running longer than provided timeout (" + str(timeout) + " seconds)")
    if remote_task.get_status() == "completed":
        return
    else:
        raise RuntimeWarning("ClearML Task did not complete correctly: " + str(remote_task.get_status()))


def generate_data(overrides: list[str]) -> tuple[Task, DictConfig]:
    """Runs datageneration with given overrides."""
    # create local config (just for validation purposes)
    with initialize(config_path="../src/ml_py_project/config/data"):
        cfg = compose(config_name="data_generation.yaml", return_hydra_config=True, overrides=overrides)

    result = subprocess.run(  # noqa: S603
        [  # noqa: S603 RUF005
            str(shutil.which("python")),  # type: ignore[list-item]
            os.path.join(os.path.dirname(__file__), "../src/ml_py_project/utils/data_generation.py"),
        ]
        + overrides,
        capture_output=True,
        text=True,
    )
    print("Results: " + str(result))
    task_id = re.findall(r"task id=(\w+)", result.stdout)[0]

    remote_task = Task.get_task(task_id=task_id)

    wait_for_task_to_complete(remote_task)

    remote_task = Task.get_task(task_id=task_id)  # update information from server

    dataset_tasks = Task.query_tasks(task_filter={"parent": remote_task.id})

    return dataset_tasks, cfg


@pytest.fixture(scope="session")
def run_default_experiment():
    """Runs the default experiment for validation."""
    validation_overrides = [
        "mlops.task_name=validation-default",
        "mlops.queue_name=H200-35GiVRAM-60GiRAM-7.5tCPU",
        "+trainer.overfit_batches=1",
        "trainer.max_epochs=10",
        "mlops.init_task=true",
        "mlops.execute_on_cluster=true",
    ]
    return run_experiment(validation_overrides)
