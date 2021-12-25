__all__ = [
    "TaiChiStep",
    "StepEnrich", "StepQuantify", "StepModeling", "StepTraining",
    "TaiChiLearn"]

from ipywidgets import HTML, IntSlider, interact
from typing import List, Callable, Any, Dict
import pandas as pd
from pathlib import Path

from tai_chi_tuna.config import PhaseConfig

from tai_chi_tuna.front.html import list_group_kv, Flash
from tai_chi_tuna.front.structure import StepByStep
from tai_chi_tuna.front.widget import interact_intercept

from tai_chi_tuna.flow.to_enrich import set_enrich, execute_enrich
from tai_chi_tuna.flow.to_quantify import (
    set_quantify, execute_quantify, TaiChiDataset, choose_xy)
from tai_chi_tuna.flow.to_model import set_datamodule, assemble_model
from tai_chi_tuna.flow.to_train import (
    make_slug_name, set_trainer, run_training)


class TaiChiStep:
    def __init__(self, name: str, progress: Dict[str, Any]):
        self.name = name
        self.progress = progress

    def __repr__(self):
        return f"Step: {self.name}"

    def assigner(self, **kwargs) -> List[str]:
        """
        Assign the values to the step object
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, **kwargs):
        """
        Call the step object
        This will be called by the StepByStep class's each step page
        """
        self.assigner(**kwargs)
        return self.action(**kwargs)

# Phase 1 - Enrichment


class StepEnrich(TaiChiStep):
    def __init__(self, progress: Dict[str, Any]):
        super().__init__("Enrich", progress)

    def action(self, **kwargs):
        set_enrich(df=self.df, phase=self.phase)

# Phase 1 - Quantify


class StepQuantify(TaiChiStep):
    def __init__(self, progress: Dict[str, Any]):
        super().__init__("Quantify", progress)

    def action(self, **kwargs):
        set_quantify(
            df=self.df,
            phase=self.phase,
            enrichments=self.enrichments_map
        )
        execute_enrich(
            df=self.df,
            phase=self.phase,
            enrichments=self.enrichments_map)

        # creating dataset
        ds = TaiChiDataset(self.df)
        self.progress['dataset'] = ds
        # preview a row of data
        display(HTML(f"<h3>A row of data</h3>"))

        @interact
        def show_row(idx=IntSlider(min=0, max=min(len(ds), 30))):
            """
            Show a row of data
            with the integer slider
            """
            list_group_kv(ds[idx])()

        choose_xy(
            progress=self.progress,
            df=self.df,
            phase=self.phase)

# Phase 3 - Modeling


class StepModeling(TaiChiStep):
    def __init__(self, progress: Dict[str, Any]):
        super().__init__("Modeling", progress)

    def action(self, **kwargs):
        qdict = execute_quantify(df=self.df, phase=self.phase,
                                 quantify_map=self.quantify_map)
        self.progress['qdict'] = qdict
        set_datamodule(self.progress, self.df, qdict, self.phase,
                       self.quantify_2_entry_map, self.quantify_2_exit_map)

# Phase 4 - Training


class StepTraining(TaiChiStep):
    def __init__(self, progress: Dict[str, Any]):
        super().__init__("Training", progress)

    def action(self, **kwargs):
        module_zoo = {"all_entry": self.all_entry, "all_exit": self.all_exit}
        final_model = assemble_model(
            self.phase, self.qdict, module_zoo)

        # save some configuration
        self.phase.save()
        task_slug = make_slug_name(self.phase)
        self.phase['task_slug'] = task_slug

        self.progress['model'] = final_model
        interact_intercept(set_trainer,
                           run_training(
                               self.phase,
                               final_model, self.datamodule)
                           )


class TaiChiLearn:
    enrichments_map = dict()
    quantify_map = dict()
    quantify_2_entry_map = dict()
    quantify_2_exit_map = dict()
    all_entry = dict()
    all_exit = dict()

    """
    A dataframe please
    then we learn
    """

    def __init__(
        self,
        df: pd.DataFrame,
        project: Path = None
    ):
        self.phase = PhaseConfig.load(project)
        self.df = df

        # setup the progress data
        self.progress = dict(
            df=self.df,
            phase=self.phase,
            enrichments_map=self.enrichments_map,
            quantify_map=self.quantify_map,
            quantify_2_entry_map=self.quantify_2_entry_map,
            quantify_2_exit_map=self.quantify_2_exit_map,
            all_entry=self.all_entry,
            all_exit=self.all_exit
        )

        # define steps
        self.steps = dict({
            "Enrich": StepEnrich(progress=self.progress),
            "Quantify": StepQuantify(progress=self.progress),
            "Model": StepModeling(progress=self.progress),
            "Train": StepTraining(progress=self.progress),
        })

        # create a step by step interactive
        self.step_by_step = StepByStep(
            self.steps, self.progress)

    def __call__(self):
        """
        display the step by step interactive
        """
        self.step_by_step()

    def __repr__(self):
        return f"{self.step_by_step} with keys:" + \
            f"{list(self.progress.keys())}"
