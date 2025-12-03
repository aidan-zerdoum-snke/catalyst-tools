# SnkeOS Internal
#
# Copyright (C) 2024-2025 Snke OS GmbH, Germany. All rights reserved.

"""Validation tests.

This file shall contain all validation tests for inference and be extended for specific projects
"""

import numpy as np
import pytest
import torch


@pytest.mark.validation(
    "This test checks if an onnx model was created as result of the training and saved in ClearML as artifact"
)
def test_onnx_model_exists(run_default_experiment):
    """Tests if an onnx model was created."""
    remote_task, _ = run_default_experiment

    artifacts = [remote_task.artifacts[art].url for art in remote_task.artifacts]

    assert any(path[-5:] == ".onnx" for path in artifacts)


@pytest.mark.validation("This test checks if the best model checkpoint matches the best metric value epoch.")
def test_output_model_optimal(run_default_experiment):
    """Tests if best output model has the correct epoch."""
    best_model_substring = "best"
    remote_task, cfg = run_default_experiment
    assert "output" in remote_task.models, f"Output model must exist, but got only {remote_task.models.keys()}."

    output_models = [output_model.get_local_copy() for output_model in remote_task.models["output"]]
    output_models = [output_model for output_model in output_models if best_model_substring in output_model]
    assert len(output_models) == 1, f"Number of best models should be exactly 1, but was {len(output_models)}."

    output_model = output_models[0]
    output_model = torch.load(output_model, weights_only=False, map_location=torch.device("cpu"))
    selected_epoch = output_model["global_step"]

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
    optimal_value_func = np.argmax if mode == "max" else np.argmin
    metric = remote_task.get_all_reported_scalars()[metric_name.split("/")[0]]["_".join(metric_name.split("/")[1:])]
    optimal_metric_value_index = optimal_value_func(metric["y"])
    optimal_epoch = metric["x"][optimal_metric_value_index] + 1  # Epoch increased before storing

    assert selected_epoch == optimal_epoch, (
        f"Epoch stored in best checkpoint ({selected_epoch}) does not match the optimal "
        f"metric value epoch ({optimal_epoch})."
    )
