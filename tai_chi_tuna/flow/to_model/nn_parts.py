__all__ = ["EntryDict", "AssembledModel", "torch", "nn", "LightningMinimal"]

from tai_chi_tuna.front.typer import LIST
from tai_chi_tuna.config import PhaseConfig
import pytorch_lightning as pl
import torch
from torch import nn
from typing import Dict


# ========== The pytorch model part ==========
class EntryDict(nn.Module):
    """
    Create entry parts for different columns
    """
    module_zoo = dict()

    def __init__(
        self,
        phase: PhaseConfig,
        qdict: Dict[str, nn.Module],
    ):
        super().__init__()
        model_dict = nn.ModuleDict()
        all_entry = self.module_zoo['all_entry']
        # only initialized x_models
        for src, model_cfg in phase['x_models'].items():
            quantify = qdict[src]

            # find column class
            model_cls = all_entry[model_cfg['model_name']]
            # the kwargs to start the column model object
            model_kwargs = model_cfg['kwargs']
            # the model object
            model = model_cls.from_quantify(quantify, **model_kwargs)

            # add the model by column name
            model_dict[src] = model

        # calculate the output size for dimention 1 (after concatenation)
        self.out_features = sum(
            list(model.out_features for src, model in model_dict.items()))
        self.model_dict = model_dict

    def forward(self, inputs):
        outputs = []
        for src, model in self.model_dict.items():
            # input data for column
            src_input = inputs[src]

            # forward pass for column_model(column_data)
            outputs.append(model(src_input))
        # concat the results
        return torch.cat(outputs, dim=1)

    @classmethod
    def update_module_zoo(cls, modules: Dict[str, Dict[str, nn.Module]]):
        """
        Assign module_zoo map to the class
        """
        if 'all_entry' not in modules:
            raise ValueError("'all_entry' model parts not in modules")
        cls.module_zoo.update(modules)
        return cls


class LightningMinimal(pl.LightningModule):
    """
    Default Mixin for lightning module
    """

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Training step in Lightning schema
        return the loss, log all the metrics (and the loss)
        """
        rt = self.loss_step(batch)
        for k, v in rt.items():
            # if the value is a scalar, log it
            if v.numel() == 1:
                self.log(f"trn_{k}", v)
        return rt['loss']

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Validation step in Lightning schema
        return the loss, log all the metrics (and the loss)
        """
        rt = self.loss_step(batch)
        for k, v in rt.items():
            # if the value is a scalar, log it
            if v.numel() == 1:
                self.log(f"val_{k}", v)
        return rt['loss']

    def configure_optimizers(self,):
        param_groups = [
            {"params": self.entry_dict.parameters(), "lr": self.entry_lr},
            {"params": self.exit_part.parameters(), "lr": self.exit_lr},
        ]
        return torch.optim.Adam(param_groups)

    @classmethod
    def update_module_zoo(cls, modules: Dict[str, Dict[str, nn.Module]]):
        """
        Assign module_zoo map to the class
        """
        if 'all_exit' not in modules:
            raise ValueError("'all_exit' model parts not in modules")
        cls.module_zoo.update(modules)
        return cls


class AssembledModel(LightningMinimal):
    """
    Create the whole parts for different columns
    """
    module_zoo = dict()

    def __init__(
        self,
        phase: PhaseConfig,
        qdict: Dict[str, nn.Module],
        entry_lr: LIST(options=[1e-1, 1e-2, 1e-3, 1e-4,
                       1e-5, 1e-6], default=1e-4) = 1e-4,
        exit_lr: LIST(options=[1e-1, 1e-2, 1e-3, 1e-4, ], default=1e-3) = 1e-3,
    ):
        super().__init__()
        self.entry_lr = entry_lr
        self.exit_lr = exit_lr
        self.entry_dict = EntryDict(phase, qdict)
        exit_cfg = list(phase['y_models'].values())[0]

        self.exit_src = exit_cfg['src']
        self.exit_kwargs = exit_cfg['kwargs']
        exit_cls = self.module_zoo['all_exit'][exit_cfg['model_name']]

        exit_quantify = qdict[self.exit_src]

        # build the exit model
        self.exit_part = exit_cls.from_quantify(
            exit_quantify, self.entry_dict, **self.exit_kwargs)

    def forward(self, inputs):
        vec = self.entry_dict(inputs)
        return self.exit_part(vec)

    def eval_forward(self, inputs):
        vec = self.entry_dict(inputs)
        return self.exit_part.eval_forward(vec)

    def loss_step(self, inputs):
        vec = self.entry_dict(inputs)
        return self.exit_part.loss_step(vec, inputs[self.exit_src])
