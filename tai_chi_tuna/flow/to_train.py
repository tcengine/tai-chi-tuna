__all__ = ["make_slug_name", "set_trainer", "run_training"]

from tai_chi_tuna.front.html import Flash
from tai_chi_tuna.front.typer import STR, INT, BOOL
from tai_chi_tuna.config import PhaseConfig
import pytorch_lightning as pl
import torch
from torch import nn
import pandas as pd
import copy
from typing import Callable, Dict, Any, List
from ipywidgets import Output
from pathlib import Path
from IPython.display import display


def make_slug_name(phase: PhaseConfig) -> str:
    xs = '-'.join(list(q['src'] for q in phase['quantify'] if q["x"]))
    ys = '-'.join(list(q['src'] for q in phase['quantify'] if q["x"] == False))
    return '_'.join([xs, 'to', ys])


class DataFrameMetricsCallback(pl.Callback):
    """
    A metrics callback keep showing pandas dataframe
    """

    def __init__(self) -> None:
        """
        In Trainer kwargs, passing this arguements along with other callbacks
        callbacks = [DataFrameMetricsCallback(),]
        """
        self.metrics: List = []

    def on_fit_start(
        self, trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        pl_module.output = Output()
        display(pl_module.output)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        self.metrics.append(dict((k, v.item())
                                 for k, v in metrics_dict.items()))
        pl_module.output.clear_output()
        with pl_module.output:
            display(pd.DataFrame(self.metrics).tail(10))


def set_trainer(
    phase,
    project: STR(default='default') = 'default',
    tensorboard: BOOL(default=True) = True,
    show_metric: BOOL(default=True) = True,
    save_top_k: INT(default=1, min_=1, max_=5) = 1,
    max_epochs: INT(min_=1, max_=200, default=5) = 5,
    use_gpu: BOOL(default=True) = True,
) -> Dict[str, Any]:
    if project == 'default':
        project = Path(phase.project)
    project = Path(project)
    task_slug = make_slug_name(phase)
    csv_logger = pl.loggers.CSVLogger(project/"csv_log", name=task_slug, )

    save_best = pl.callbacks.ModelCheckpoint(
        dirpath=project/"checkpoints",
        filename='{epoch}-{val_loss:.2f}.ckpt',
        save_top_k=save_top_k,
        save_weights_only=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    loggers = [
        csv_logger
    ]
    if tensorboard:
        loggers.append(
            pl.loggers.TensorBoardLogger(
                save_dir=project/'tensorboard',
                name=task_slug)
        )
    rt = dict(
        max_epochs=max_epochs,
        logger=loggers)
    callbacks = [save_best]
    if show_metric:
        callbacks.append(
            DataFrameMetricsCallback())

    rt.update({"callbacks": callbacks})

    if use_gpu:
        if torch.cuda.is_available():
            rt.update(dict(gpus=1))
        else:
            Flash.warning(
                "GPU is not available, use CPU instead", key="Warning")
            rt.update(dict(gpus=0))
    return rt


def run_training(
    phase: PhaseConfig,
    final_model: nn.Module,
    datamodule: pl.LightningDataModule
) -> Callable:
    def set_trainer_callback(kwargs) -> pl.Trainer:
        task_slug = phase['task_slug']

        # create trainer
        Flash.info(
            f"Create trainer for task: {task_slug}", key="Notice")
        trainer_kwargs = set_trainer(
            phase, **kwargs)
        trainer = pl.Trainer(**trainer_kwargs)

        # start training
        Flash.success(
            "Start training, this is not a drill!", key="Alert!")

        phase.save()
        trainer.fit(final_model, datamodule=datamodule)
        return trainer
    return set_trainer_callback
