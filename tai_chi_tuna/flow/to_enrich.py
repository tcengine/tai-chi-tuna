__all__ = ["set_enrich", "execute_enrich"]

from tai_chi_tuna.front.structure import EditableList
from tai_chi_tuna.front.html import DOM
from tai_chi_tuna.front.widget import (
    init_interact, reconfig_manual_interact
)
from tai_chi_tuna.config import PhaseConfig
from ipywidgets import HTML, interact_manual
from typing import Dict, Any
from tqdm.notebook import tqdm
import pandas as pd
from IPython.display import display

def set_enrich(**kwargs) -> None:
    df = kwargs['df']
    phase = kwargs['phase']
    ENRICHMENTS = kwargs['enrichments_map']

    DOM(f"{len(df)} rows of data, example table", "h3")()
    display(df.sample(5))
    display(HTML("<hr>"))

    def setting_col():
        enrich_data_list = phase['enrich'] if 'enrich' in phase else []
        enrich_box = EditableList(enrich_data_list)
        display(enrich_box)

        def set_enrich_(src=["[all_columns]", ]+list(df.columns)):
            DOM(f"Setting up column enrich: {src}", "h4")()
            if src == "[all_columns]":
                display(df.head(3))
            else:
                display(df[[src, ]].head(3))

            def choose_enrich(dst="", enrich=ENRICHMENTS):
                DOM(f"Source: {src}, Destination: {dst}, for {enrich.__name__}", "h4")(
                )
                DOM(f"{enrich.__doc__}", "div")()

                def result_callback(kwargs: Dict[str, Any]) -> None:
                    """
                    A callback function to put in new row of data
                    into the editable list
                    """
                    # the current config data
                    extra = {
                        "src": src,
                        "dst": dst,
                        "kwargs": kwargs,
                        "enrich": enrich.__name__}
                    # put to editable list
                    enrich_box+extra
                    # set to phase_config
                    phase['enrich'] = enrich_box.get_data()
                obj, decoed_func = init_interact(enrich, result_callback)
            choose_enrich_widget = interact_manual(choose_enrich).widget
            reconfig_manual_interact(
                choose_enrich_widget,
                description="Choose",
                button_style='warning'
            )
        set_enrich_widget = interact_manual(set_enrich_).widget
        reconfig_manual_interact(set_enrich_widget, button_style='warning')
    setting_col()

def execute_enrich(
    df: pd.DataFrame,
    phase:PhaseConfig,
    enrichments: Dict[str,Any]
) -> pd.DataFrame:
    """
    Execute enrichments
    takes a dataframe, enrichments and phase config
    returns the same dataframe with enriched columns
    """
    if 'enrich' not in phase:
        return df
    for en_conf in tqdm(phase["enrich"], leave=False):
        enrich_name = en_conf['enrich']
        enrich_cls = enrichments[enrich_name]
        kwargs = en_conf['kwargs']
        src = en_conf['src']
        dst = en_conf['dst']
        # The class with lazy loading, will only 
        # call the class only if necessary
        if enrich_cls.lazy:
            obj = enrich_cls(**kwargs)
            obj.src = src
            df[dst] = obj
        # The class without lazy loading
        # create the column now
        else:
            obj = enrich_cls(**kwargs)
            if src=="[all_columns]":
                df[dst] = df.apply(obj, axis=1)
            else:
                df[dst] = df[src].apply(obj)
    return df
