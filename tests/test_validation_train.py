# SnkeOS Internal
#
# Copyright (C) 2024-2025 Snke OS GmbH, Germany. All rights reserved.

"""Validation tests.

This file shall contain all validation tests for training and be extended for specific projects
"""

import subprocess

import numpy as np
import pytest
from omegaconf import OmegaConf

from tests.conftest import run_experiment


@pytest.mark.validation(
    "Performs an overfitting training: Executes a training run with default parameters, but the size of the train and "
    "dev dataset is reduced to 1 and the number of epochs is reduced to 10. Checks if no error is thrown."
)
def test_run_default_experiment(run_default_experiment):
    """Executes a default experiment and checks if it starts without errors."""
    _, _ = run_default_experiment


@pytest.mark.validation(
    "This test checks if a Task was generated in ClearML that has the same name as provided in the config"
)
def test_clearml_task_exists(run_default_experiment):
    """Tests if the clearML task exists."""
    remote_task, cfg = run_default_experiment

    # check that task exists
    assert remote_task is not None

    # check that name is same than in config
    assert remote_task.name == cfg.mlops.task_name


@pytest.mark.validation(
    "This test checks if the code version (git commit id) of the local code and executed code on clearML are identical"
)
def test_correct_code_used(run_default_experiment):
    """Tests if the correct code is used."""
    remote_task, _ = run_default_experiment

    # check that clearML uses the same code version
    clearml_git_commit_id = remote_task.data.script.version_num

    local_git_commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()  # noqa: S603, S607

    assert clearml_git_commit_id == local_git_commit_id


@pytest.mark.validation(
    "This test verifies that the training process has been correctly executed, by verifying that the number of "
    "training epochs equals the number of epochs set in the input configuration file (=10)."
)
def test_number_of_epochs_is_correct(run_default_experiment):
    """Tests if the number of epochs is indeed 10."""
    remote_task, cfg = run_default_experiment

    loss_curve = remote_task.get_all_reported_scalars()["train"]["loss_epoch"]["y"]

    assert len(loss_curve) == cfg.trainer.max_epochs


@pytest.mark.validation("This test verifies that the training process logs the specified metrics.")
def test_all_metrics_logged(run_default_experiment):
    """Tests if metrics logged."""
    remote_task, cfg = run_default_experiment
    metrics_dev = [metric["name"] for metric in cfg["model"]["metrics"] if "dev" in metric["run_on_dataloaders"]]
    metrics_train = [metric["name"] for metric in cfg["model"]["metrics"] if "train" in metric["run_on_dataloaders"]]

    # Depending on whether the metric returns a dict or a scalar, the value is either reported under an individual
    # string (e.g., `metric['dev_ClassDice'])` or in the respective phase (e.g., `metric['dev']['Dice']`). Thus the `or`
    # comparison here.
    assert all(
        (
            f"dev_{metric_dev}" in remote_task.get_all_reported_scalars()
            or metric_dev in remote_task.get_all_reported_scalars()["dev"]
        )
        for metric_dev in metrics_dev
    ), "Could not find all dev metrics specified in config in reported values."
    assert all(
        (
            f"train_{metric_train}" in remote_task.get_all_reported_scalars()
            or metric_train in remote_task.get_all_reported_scalars()["train"]
        )
        for metric_train in metrics_train
    ), "Could not find all train metrics specified in config in reported values."


@pytest.mark.validation("This test checks if the loss of for the training run went down over several epochs")
def test_experiment_loss_decreases(run_default_experiment):
    """Tests if the loss decreases."""
    remote_task, _ = run_default_experiment

    loss_curve = remote_task.get_all_reported_scalars()["train"]["loss_epoch"]["y"]
    assert loss_curve[-1] < loss_curve[0]


@pytest.mark.validation(
    "Initializes a second experiment with changed parameters (learning rate, unet filters) and verifies that these"
    "changes are transfered to clearML."
)
def test_run_default_parameter_experiment():
    """Executes an experiment with changed parameters and checks if these are visible in clearML."""
    validation_overrides = [
        "mlops.task_name=validation-parameter-change",
        "mlops.queue_name=",  # set queue to none such that the experiment is not executed
        "mlops.init_task=true",
        "mlops.execute_on_cluster=true",
        "model.optimizer.lr=1e-3",
        "model.model.net.channels=[4,8,16]",
    ]

    remote_task, _ = run_experiment(validation_overrides)

    remote_conf = OmegaConf.create(remote_task.data.configuration["OmegaConf"].value)

    assert remote_conf.model.optimizer.lr == 1e-3
    assert remote_conf.model.model.net.channels == [4, 8, 16]


@pytest.mark.validation("This test checks if the number of trainable parameters is logged.")
def test_experiment_output_parameters_logged(run_default_experiment):
    """Tests if number of parameters logged."""
    remote_task, _ = run_default_experiment

    console_output = remote_task.get_reported_console_output(20)
    assert any("Trainable params" in line for line in console_output)
    assert any("Non-trainable params" in line for line in console_output)
    assert any("Total params" in line for line in console_output)


@pytest.mark.validation(
    "Initializes a second experiment with increased unet filters and verifies that the number of total parameters"
    "increased."
)
def test_run_change_parameter_count():
    """Executes an experiment with more parameters and checks if the count increased in the task."""
    validation_overrides_normal_parameters = [
        "mlops.task_name=validation-parameter-change",
        "mlops.queue_name=H200-35GiVRAM-60GiRAM-7.5tCPU",
        "+trainer.overfit_batches=1",
        "trainer.max_epochs=10",
        "mlops.init_task=true",
        "mlops.execute_on_cluster=true",
        "model.model.net.channels=[2,4,8]",
    ]
    validation_overrides_increased_parameters = [
        "mlops.task_name=validation-parameter-change",
        "mlops.queue_name=H200-35GiVRAM-60GiRAM-7.5tCPU",
        "+trainer.overfit_batches=1",
        "trainer.max_epochs=10",
        "mlops.init_task=true",
        "mlops.execute_on_cluster=true",
        "model.model.net.channels=[4,8,16]",
    ]

    def get_parameter_count_from_line(line):
        count = line.split("Total params")[0].split("\n")[-1].split(" ")[0]
        return float(count)

    remote_task_normal, _ = run_experiment(validation_overrides_normal_parameters)
    remote_task_increased, _ = run_experiment(validation_overrides_increased_parameters)

    console_output_normal_task = remote_task_normal.get_reported_console_output(200)
    console_output_increased_task = remote_task_increased.get_reported_console_output(200)

    normal_parameter_count = [
        get_parameter_count_from_line(line) for line in console_output_normal_task if "Total params" in line
    ]
    increased_parameter_count = [
        get_parameter_count_from_line(line) for line in console_output_increased_task if "Total params" in line
    ]
    assert len(normal_parameter_count) == 1 or len(increased_parameter_count) == 1
    assert normal_parameter_count[0] < increased_parameter_count[0]


@pytest.mark.validation("This test checks if validation metric improves from first to best epoch.")
def test_metric_improves():
    """Tests if best metric determined by the callback is better than the first value."""
    validation_overrides_parameters = [
        "mlops.task_name=validation",
        "mlops.queue_name=H200-35GiVRAM-60GiRAM-7.5tCPU",
        "+trainer.overfit_batches=1",
        "trainer.max_epochs=10",
        "trainer.check_val_every_n_epoch=1",
        "mlops.init_task=true",
        "mlops.execute_on_cluster=true",
    ]
    remote_task, cfg = run_experiment(validation_overrides_parameters)

    best_model_callbacks = [
        callback
        for callback in cfg.callbacks.values()
        if "ModelCheckpoint" in callback["_target_"] and "monitor" in callback
    ]
    assert len(best_model_callbacks) == 1, (
        f"Only one best_model_callback should exists, but found {len(best_model_callbacks)}."
    )
    best_model_callback = best_model_callbacks[0]
    metric_name = best_model_callback["monitor"]
    mode = best_model_callback["mode"]
    optimal_value_func = np.max if mode == "max" else np.min
    metric = remote_task.get_all_reported_scalars()[metric_name.split("/")[0]][metric_name.split("/")[1]]
    optimal_metric_value: float = optimal_value_func(metric["y"])
    first_metric_value: float = metric["y"][0]

    assert optimal_metric_value > first_metric_value if mode == "max" else optimal_metric_value < first_metric_value, (
        f"First epoch metric value was {first_metric_value} and the best model metric value was "
        f"{optimal_metric_value}, which is not an improvement."
    )
