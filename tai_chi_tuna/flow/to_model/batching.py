__all__ = ["TaiChiCollate", "TaiChiDataModule",]

from tai_chi_tuna.flow.to_quantify import TaiChiDataset
from tai_chi_tuna.front.typer import FLOAT, LIST
import pytorch_lightning as pl
import torch
from typing import Dict, Any
import pandas as pd

class TaiChiCollate:
    """
    Universal all power full collate function
    1 for all collation
    """

    def __init__(self, quantify_dict):
        self.quantify_dict = quantify_dict

    def make_df(self, batch):
        return pd.DataFrame(list(batch))

    def __len__(self):
        return len(self.quantify_dict)

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        """
        This call will execute the __call__(a_list_of_items)
        from Quantify objects column by column
        """
        batch_df = self.make_df(batch)
        rt = dict()
        for src, qobj in self.quantify_dict.items():
            rt.update({
                src: qobj(list(batch_df[src]))
            })
        return rt


class TaiChiDataModule(pl.LightningDataModule):
    """
    Universal all power full lightning data module
    """

    def __init__(self, dataset: TaiChiDataset, quantify_dict: Dict[str, Any]):
        super().__init__()
        self.dataset = dataset
        self.quantify_dict = quantify_dict

        self.collate = TaiChiCollate(quantify_dict)

    def configure(
        self,
        valid_ratio: FLOAT(min_=0.01, max_=0.5, default=.1, step=0.01) = .1,
        batch_size: LIST(options=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], default=32) = 32,
        shuffle: LIST(options=[True, False], default=False) = True,
        num_workers: LIST(options=[0, 2, 4, 8, 16], default=0) = 0,
    ):
        """
        Configure the data module
        usually read thest from the "batch_level" key of
            phase_config
        """
        self.train_ds, self.val_ds = self.dataset.split(valid_ratio)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        self.train_dl = self.train_ds.dataloader(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last = True,
            )
        self.train_dl.collate_fn = self.collate
        return self.train_dl

    def val_dataloader(self):
        self.val_dl = self.val_ds.dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers, 
            drop_last = False,
            )
        self.val_dl.collate_fn = self.collate
        return self.val_dl