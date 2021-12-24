
from tai_chi_tuna.flow.to_quantify import TaiChiDataset
from tai_chi_tuna.front.html import Flash
from tai_chi_tuna.front.typer import FLOAT, LIST
from tai_chi_tuna.front.structure import EditableDict
from tai_chi_tuna.front.widget import (
    InteractiveAnnotations,
    interact_intercept,
    reconfig_manual_interact)
from tai_chi_tuna.config import PhaseConfig
import pytorch_lightning as pl
import torch
from typing import Dict, Any
import pandas as pd
from ipywidgets import HTML, Button, Output, interact_manual

# batching data part of the modeling
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
        for src,qobj in self.quantify_dict.items():
            rt.update({
                src:qobj(list(batch_df[src]))
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
        valid_ratio:FLOAT(min_=0.01, max_=0.5, default=.1, step=0.01)=.1,
        batch_size: LIST(options=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], default=32) = 32,
        shuffle: LIST(options=[True, False], default=False) = True,
        num_workers: LIST(options=[0, 2, 4, 8, 16], default=0) =0,
    ):  
        """
        Configure the data module
        usually read thest from the "batch_level" key of
            phase_config
        """
        self.train_ds, self.val_ds = self.dataset.split(valid_ratio)
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.num_workers=num_workers
        
    def train_dataloader(self):
        self.train_dl = self.train_ds.dataloader(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers)
        self.train_dl.collate_fn = self.collate
        return self.train_dl
    
    def val_dataloader(self):
        self.val_dl = self.val_ds.dataloader(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers)
        self.val_dl.collate_fn = self.collate
        return self.val_dl

def choose_models(
    quantify,
    cls_options,
    model_conf: EditableDict,
):
    def config_model(ModelClass=cls_options):
        def starting_cls(kwargs):
            model_conf[quantify.src] = dict(
                model_name=ModelClass.__name__,
                src=quantify.src,
                kwargs=kwargs,
            )

        ia = InteractiveAnnotations(
            ModelClass.from_quantify,
            description="Okay",
            icon='rocket',
            button_style='success')

        ia.register_callback(starting_cls)
        display(ia.vbox)
    inter = interact_manual(config_model)
    reconfig_manual_interact(
        inter.widget,
        description="Yes!", icon="cube", button_style='info')
    return inter


def set_model(
    quantify_dict: Dict[str, Any],
    phase: PhaseConfig, 
    quantify_2_entry_map: Dict[str, Any],
    quantify_2_exit_map: Dict[str, Any],
    ):
    display(HTML("""<h3>Set up model structure</h3>
    <quote>You'll have to setup a model part for each of the column</quote>"""))

    x_models = EditableDict()
    y_models = EditableDict()

    if "x_models" in phase:
        x_models + phase['x_models']
    if "y_models" in phase:
        y_models + phase['y_models']
    display(HTML("<h3>Current model config:</h3>"))
    display(HTML(f"""
    <h3 class='text-primary'>🤖 <strong>Entry</strong> parts of the model</h3>
    <h4>To understand the X columns</h4>
    """))
    display(x_models)
    display(HTML(f"""
    <h3 class='text-danger'>🦾 <strong>Exit</strong> parts of the model</h3>
    <h4>To understand & predict the Y column</h4>
    """))
    display(y_models)

    @x_models.on_update
    def update_x_models(x_models_data):
        phase['x_models'] = x_models_data

    @y_models.on_update
    def update_y_models(y_models_data):
        phase['y_models'] = y_models_data

    display(HTML("<h4>Change model config:</h4>"))
    for src, quantify in quantify_dict.items():
        if quantify.is_x:
            entry_cls_options = dict(
                (q.__name__, q)
                for q in quantify_2_entry_map.get(quantify.__class__))

            if entry_cls_options is None:
                Flash.danger(
                    f"We do not support {quantify.__class__} as X data",
                    key="Error!")
                continue
            display(HTML(f"""
            <h3 class='text-primary'>Choose Model For X Columns:
            <strong>{src}</strong></h3>"""))
            choose_models(quantify, entry_cls_options, x_models)
    for src, quantify in quantify_dict.items():
        if quantify.is_x == False:
            exit_cls_options = dict(
                (q.__name__, q)
                for q in quantify_2_exit_map.get(quantify.__class__))
            if entry_cls_options is None:
                Flash.danger(
                    f"We do not support {quantify.__class__} as Y data",
                    key="Error!"
                )
            display(HTML(f"""
            <h3 class='text-danger'>Choose Model For Y Column:
            <strong>{src}</strong></h3>"""))
            choose_models(quantify, exit_cls_options, y_models)


def set_datamodule(progress, df, qdict, phase,
    quantify_2_entry_map: Dict[str, Any],
    quantify_2_exit_map: Dict[str, Any],
):
    # intitalize the dataset and datamodule at this step
    ds = TaiChiDataset(df)
    datamodule = TaiChiDataModule(ds, qdict)

    batch_level = EditableDict()
    if "batch_level" in phase:
        batch_level['batch_level'] = phase['batch_level']
    display(HTML("<h3>How we make data rows into batch</h3>"))
    display(batch_level)
    model_output = Output()

    def configure_setting(kwargs):
        batch_level['batch_level'] = kwargs
        phase['batch_level'] = kwargs
        set_model_btn_event()

    def set_model_btn_event():
        if 'batch_level' not in phase:
            Flash.warning("batch level config not set",
                          key="Warning")
            return
        datamodule.configure(**phase['batch_level'])
        progress.kwargs['datamodule'] = datamodule

        model_output.clear_output()
        with model_output:
            set_model(
                qdict, phase,
                quantify_2_entry_map,
                quantify_2_exit_map)

    interact_intercept(datamodule.configure, configure_setting)

    set_model_btn = Button(description="Set Batch",
                           icon='cog', button_style='info')
    set_model_btn.click = set_model_btn_event
    display(set_model_btn)
    display(model_output)