__all__ = [
    "TaiChiStep",
    "StepEnrich", "StepQuantify", "StepModeling", "StepTraining",
    "TaiChiLearn"]

from ipywidgets import HTML, IntSlider, interact
from typing import List, Callable, Any, Dict
import pandas as pd
from pathlib import Path
# configuration
from tai_chi_tuna.config import PhaseConfig

# frontend & interactive
from tai_chi_tuna.front.html import list_group_kv, Flash
from tai_chi_tuna.front.structure import StepByStep
from tai_chi_tuna.front.widget import interact_intercept

# workflow
from tai_chi_tuna.flow.to_enrich import set_enrich, execute_enrich
from tai_chi_tuna.flow.to_quantify import (
    execute_quantify, TaiChiDataset, choose_xy,
    save_qdict, load_qdict
)

# modeling
from tai_chi_tuna.flow.to_model import (
    set_datamodule, assemble_model, set_opt_confs, ParamWizard
)

#training
from tai_chi_tuna.flow.to_train import (
    make_slug_name, set_trainer, run_training)

from tai_chi_tuna.utils import clean_name
from IPython.display import display


class TaiChiStep:
    """
    This is a concept of the step object
    Where a single step can be defined here,
        and going back to if necessary
    Will execute the class function action(self, **kwargs)
    When the object is called as a method
    """
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
    """
    Enrichment Step
    Where user can add more dataframe columns
    The step can be defined and automated later
    """

    def __init__(self, progress: Dict[str, Any]):
        super().__init__("Enrich", progress)

    def action(self, **kwargs):
        set_enrich(
            df=self.df, phase=self.phase,
            enrichments_map=self.enrichments_map
        )

# Phase 1 - Quantify


class StepQuantify(TaiChiStep):
    """
    Quantify: The step where the user can:
    1. Choose the x and y columns, there could be multiple x columns
    2. Choose how the columns' data can be quantify to tensors
    """
    def __init__(self, progress: Dict[str, Any]):
        super().__init__("Quantify", progress)

    def action(self, **kwargs):
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
            **self.progress)

# Phase 3 - Modeling


class StepModeling(TaiChiStep):
    """
    This step defines how we create a deep learning model
    """
    def __init__(self, progress: Dict[str, Any]):
        super().__init__("Modeling", progress)

    def action(self, **kwargs):
        # bring configuration of quantify details
        # to actual python objects
        qdict = execute_quantify(df=self.df, phase=self.phase,
                                 quantify_map=self.quantify_map)
        save_qdict(self.phase.project, qdict)

        # save the qdict to the progress
        self.progress['qdict'] = qdict

        set_datamodule(self.progress, self.df, qdict, self.phase,
                       self.quantify_2_entry_map, self.quantify_2_exit_map)

# Phase 4 - Training


class StepTraining(TaiChiStep):
    def __init__(self, progress: Dict[str, Any]):
        super().__init__("Training", progress)

    def action(self, **kwargs):
        module_zoo = {"all_entry": self.all_entry, "all_exit": self.all_exit}

        # assemble a pytorch model
        Flash.info(f"Creating final model, takes time...", key="ALERT")
        final_model = assemble_model(
            self.phase, self.qdict, module_zoo)

        # save some configuration
        self.phase.save()

        self.param_wizard = ParamWizard(final_model)
        self.param_wizard.set_configure_optimizers(self.phase)

        set_opt_confs(self.param_wizard, self.phase)

        task_slug = make_slug_name(self.phase)
        self.phase['task_slug'] = task_slug

        self.progress['model'] = final_model
        # create a training function based on 
        training_function = run_training(
            self.phase, # configuration
            final_model, # pytroch model
            self.datamodule) # data pipeline

        interact_intercept(set_trainer,
                           training_function)


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
        project = Path(project)
        self.phase = PhaseConfig.load(project)
        # this is a very strange step to step attribute outside of object
        self.phase.project = project
        self.df = df
        # clean dirty column names
        self.df = self.df.rename(columns=dict((col, clean_name(col))
                                              for col in self.df.columns))

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

        # chain up all the steps
        self.steps = dict({
            "Enrich": StepEnrich(progress=self.progress),
            "Quantify": StepQuantify(progress=self.progress),
            "Model": StepModeling(progress=self.progress),
            "Train": StepTraining(progress=self.progress),
        })

    def __call__(self):
        """
        display the step by step interactive
        """
        # create a step by step interactive
        self.step_by_step = StepByStep(
            self.steps, kwargs=self.progress)
        self.step_by_step()

    def __repr__(self):
        if hasattr(self, 'step_by_step'):
            return f"{self.step_by_step} with keys:\n" + \
                f"{list(self.progress.keys())}"
        else:
            return f"Engine on project:{self.phase.project}"
