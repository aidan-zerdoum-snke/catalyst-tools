# SnkeOS Internal
#
# Copyright (C) 2024-2025 Snke OS GmbH, Germany. All rights reserved.

"""Minimal example main."""

import hydra
import lightning as pl
import rootutils
from clearml import Task
from snketorch.config.common import RootConfig

from snketorch.utils import utils
from snketorch.clearml.task import clearml_task_init
from snketorch.clearml.model import load_model_checkpoint
from snketorch.utils.pylogger import get_pylogger
from snketorch.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_model,
)

# this will activate omega_conf resolvers, i.e., make it possible to use variables in hydra configs
utils.init_extras()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# activate logger
log = get_pylogger(__name__)

# baseline configs from snketorch
utils.register_configs()


@hydra.main(version_base="1.3", config_path="./config", config_name="config.yaml")
def main(cfg: RootConfig) -> None:
    """Trains the model."""
    # set seed for random number generators in pytorch, numpy and python.random
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.mlops.init_task:
        task = clearml_task_init(cfg.mlops)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data, _convert_="object")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate_model(cfg)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, _convert_="object")

    log.info("Starting TRAINING")

    # Load last checkpoint in case this execution continues an interrupted task
    ckpt_path = None
    if task := Task.current_task():
        if task.get_models().get("output"):
            ckpt_path = load_model_checkpoint()
        else:
            log.info("Model will start training with no pre-loaded weights")
    else:
        log.info("Model will start training with no pre-loaded weights")

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    # run evaluation (necessary for onnx export)
    log.info("Starting task EVALUATION")
    if task := Task.current_task():
        model_weights_artifact = task.models["output"][cfg.callbacks.model_checkpoint.filename]
        ckpt_path = model_weights_artifact.get_local_copy()
    else:
        ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info(f"Best ckpt path: {ckpt_path}")


if __name__ == "__main__":
    main()
