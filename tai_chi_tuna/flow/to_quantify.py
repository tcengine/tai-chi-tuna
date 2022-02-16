__all__ = [
    "TaiChiDataset", "choose_xy", "execute_quantify",
    "SIZE_DIMENSION", "BATCH_SIZE", "SEQUENCE_SIZE",
    "IMAGE_SIZE", "save_qdict", "load_qdict",
]

from tai_chi_tuna.front.html import DOM
from tai_chi_tuna.front.typer import FLOAT, LIST
from tai_chi_tuna.front.structure import EditableList
from tai_chi_tuna.front.widget import init_interact
from tai_chi_tuna.config import PhaseConfig
from ipywidgets import HTML, Dropdown, interact_manual
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from IPython.display import display


class SIZE_DIMENSION:
    pass


class BATCH_SIZE(SIZE_DIMENSION):
    pass


class SEQUENCE_SIZE(SIZE_DIMENSION):
    pass


class IMAGE_SIZE(SIZE_DIMENSION):
    pass


class TaiChiDataset(Dataset):
    """
    A pytorch dataset working under our core engine
    The dataset class should on be defined here once
    """

    def __init__(self, df, columns: List[Any] = None):
        self.df = df
        self.columns = list(df.columns) if columns is None else columns

    def __len__(self) -> int:
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1.).reset_index(drop=True)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Universal getitem function
        """
        row = dict(self.df.loc[idx])
        rt = dict()
        for col in self.columns:
            v = row[col]
            if hasattr(v, "is_enrich"):
                rt[col] = v.rowing(row)
            else:
                rt[col] = v
        return rt

    def split(
        self,
        valid_ratio: FLOAT(min_=0.01, max_=0.5, default=.1, step=0.01) = .1
    ) -> Tuple[Any]:
        """
        Split dataset to train, validation
        """
        cls = self.__class__
        slicing = (np.random.rand(len(self)) < valid_ratio)
        return (
            cls(self.df[~slicing].reset_index(drop=True), self.columns),
            cls(self.df[slicing].reset_index(drop=True), self.columns)
        )

    def dataloader(
        self,
        batch_size: LIST(options=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], default=32) = 32,
        shuffle: LIST(options=[True, False], default=False) = False,
        num_workers: LIST(options=[0, 2, 4, 8, 16], default=0) = 0,
        drop_last = False,
    ) -> DataLoader:
        """
        Create dataloader from dataset
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            )


def choose_xy(**kwargs):
    df = kwargs.get("df")
    phase = kwargs.get("phase")
    ENRICHMENTS = kwargs.get("enrichments_map")
    QUANTIFY = kwargs.get("quantify_map")

    DOM(f"{len(df)} rows of data, example table", "h3")()
    display(df.sample(5))
    display(HTML("<hr>"))
    DOM("Please Choose Column", "h3")()
    DOM("The AI model will try to guess the Y with the input X",
        "div", {"style": "color:#666699"})()

    task = 'quantify'
    # enrich by columns
    if "enrich" in phase:
        by_destination = dict((en['dst'], en) for en in phase['enrich'])
    else:
        by_destination = dict()

    data_list = phase[task] if task in phase else []
    quantify_list = EditableList(data_list)
    display(quantify_list)

    @interact_manual
    def set_quantify_(src=list(df.columns), use_for=["As X", "As Y"]):
        DOM(f"Quantify Column: {src} {use_for}", "h4")()
        display(df[[src, ]].head(3))

        quantify_dropdown = Dropdown(options=list(QUANTIFY.keys()))

        # check the hint from last step
        # if we enrich the column as image,
        # we would certainly prefer QuantifyImage as Quantifying step
        prefer = None
        if src in by_destination:
            col_config = by_destination[src]
            cls = ENRICHMENTS[col_config['enrich']]

            # In case the enrich layer has the preference
            if hasattr(cls, "prefer"):
                prefer = cls.prefer

                # set default value to drop down value,
                # if the the previous hint suggest so
                quantify_dropdown.value = prefer
                DOM(f"Prefered quantifying:\t{cls.prefer}", "h4")()
            if hasattr(cls, "typing"):
                DOM(f"Output data type:\t{cls.typing}", "h4")()

        @interact_manual
        def choose_quantify(quantify=quantify_dropdown):
            # get the quantify class
            cls = QUANTIFY[quantify]

            def result_callback(kwargs):
                # new configuration about quantify
                extra = {"src": src, "x": (use_for == "As X"),
                         "kwargs": kwargs, "quantify": cls.__name__}
                # add the new config to editable list
                quantify_list+extra
                # update the editable list to phase
                phase['quantify'] = quantify_list.get_data()

            obj, decoded = init_interact(cls, result_callback)

def save_qdict(project:Path, qdict: Dict[str, Any]):
    """
    Save the quantify dict to phase
    """
    project = Path(project)
    project.mkdir(exist_ok=True)
    for name, quantify in qdict.items():
        quantify.save(project, name)
    return project/"quantify"

def load_qdict(
    project:Path, phase: PhaseConfig, quantify_map: Dict[str, Any]
    ) -> Dict[str, Any]:
    """
    Load the quantify dict from phase and disk saved info
    """
    project = Path(project)
    qdict = dict()
    for quant_conf in phase['quantify']:
        quantify_cls_name = quant_conf['quantify']
        quantify_cls = quantify_map[quantify_cls_name]
        name = quant_conf['src']
        # initialize the quantify object
        qobj = quantify_cls.load(project, name)
        qobj.phase = phase
        qobj.is_inference = True
        qobj.src = quant_conf['src']
        qobj.is_x = quant_conf['x']
        qdict[name] = qobj
    return qdict

def execute_quantify(
    df: pd.DataFrame, phase: PhaseConfig,
    quantify_map: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the quantify phase
    """
    # existance check
    if 'quantify' not in phase:
        raise KeyError(f"No quantify stepset")

    qdict = dict()
    for i, qconf in tqdm(enumerate(phase['quantify']), leave=False):
        qname = qconf['quantify']
        kwargs = qconf['kwargs']
        src = qconf['src']
        x = qconf['x']

        cls = quantify_map[qname]
        # initialize the quantify class
        qobj = cls(**kwargs)
        qobj.phase = phase
        qobj.is_inference = False
        qobj.src = src
        qobj.is_x = x
        qobj.adapt(df[src])
        qdict.update({src: qobj})
    return qdict


