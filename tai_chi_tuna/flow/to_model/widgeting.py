__all__ = ["choose_models",
           "set_model", "set_datamodule",
           "assemble_model", "set_opt_confs"]

from tai_chi_tuna.flow.to_quantify import TaiChiDataset
from tai_chi_tuna.front.html import Flash
from tai_chi_tuna.front.structure import EditableDict
from tai_chi_tuna.front.widget import (
    InteractiveAnnotations,
    interact_intercept,
    reconfig_manual_interact)
from tai_chi_tuna.front.typer import LIST, FLOAT, BOOL
from tai_chi_tuna.config import PhaseConfig
from typing import Dict, Any
from ipywidgets import (
    HTML, Button, HBox,
    Output, interact_manual,
    interact, Dropdown
)
import logging

from .batching import TaiChiDataModule
from .nn_parts import (nn, EntryDict, AssembledModel)
from .optimizing import ParamWizard
from IPython.display import display


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
        progress['datamodule'] = datamodule

        model_output.clear_output()
        with model_output:
            set_model(
                qdict, phase,
                quantify_2_entry_map,
                quantify_2_exit_map)

    interact_intercept(datamodule.configure, configure_setting)
    set_model_btn_event()
    # set_model_btn = Button(description="Set Batch",
    #                        icon='cog', button_style='info')
    # set_model_btn.click = set_model_btn_event
    # display(set_model_btn)
    display(model_output)


def assemble_model(
    phase: PhaseConfig,
    qdict: Dict[str, Any],
    modules: Dict[str, Dict[str, nn.Module]],
) -> nn.Module:
    logging.info("Assemble model, takes time")
    EntryDict.update_module_zoo(modules)
    if "y_models" in phase:
        y_models = phase["y_models"]
        if len(y_models) > 1:
            raise ValueError("Multiple targets are not supported by now")
        else:
            AssembledModel.update_module_zoo(modules)
            return AssembledModel(phase, qdict)
    else:
        logging.warning("No target model is specified")
        raise ValueError("phase must contain 'y_models' configuration for now")


# optimizer configuration
def optimizer_group_conf(
    freeze: BOOL(default=False),
    lr: LIST(options=list(
        f'1e-{i}' for i in range(1, 8)), default="1e-3") = "1e-3",
    weight_decay: FLOAT(min_=0., max_=.3, default=.0, step=.01) = 0.
):
    return dict(lr=lr, weight_decay=weight_decay)


def combine_prefix(prefix: str, sub: str) -> str:
    if prefix != "":
        return f"{prefix}.{sub}"
    else:
        return sub


def set_opt_confs(wizard: ParamWizard, phase: PhaseConfig):
    if 'param_groups' in phase:
        editable = EditableDict(data_dict = phase['param_groups'])
    else:
        editable = EditableDict()

    @editable.on_update
    def set_phase(kwargs):
        phase['param_groups'] = kwargs

    def create_conf_ia_callback(prefix, sub, kw_drop):
        def set_conf(conf):
            editable[f"{combine_prefix(prefix, sub)}|{kw_drop.value}"] = conf
        return set_conf

    def deeper_click(prefix, output):
        def wr():
            output.clear_output()
            with output:
                return find_next(prefix)
        return wr

    def btn_conf_widget(prefix, sub, output, kw_drop):
        def btn_conf_widget_click():
            output.clear_output()
            with output:
                title = HTML(
                    f"""<h4>
                        Set Learning for
                        <strong class='text-danger'>
                        "{combine_prefix(prefix, sub)}"</strong></h4>""")
                conf_banner = HBox([
                    title, kw_drop
                ])
                display(
                    conf_banner
                )
                ia = InteractiveAnnotations(
                    optimizer_group_conf,
                    icon="terminal", description="Yes"
                )

                ia.register_callback(
                    create_conf_ia_callback(prefix, sub, kw_drop)
                )
                display(ia.vbox)
        return btn_conf_widget_click

    def control_layer(prefix, sub, ct):
        sub_hbox_list = []
        output = None
        if ct >= 2:
            deeper = Button(button_style="success",
                            icon="plus", description="Deeper")
            # go deeper recursively
            output = Output(
                layout={
                    "border": "2px dashed #7780FF",
                }
            )
            deeper.click = deeper_click(combine_prefix(prefix, sub), output)
            sub_hbox_list.append(deeper)

        kw_df = wizard.find_kw(prefix=combine_prefix(prefix, sub))
        kw_drop = Dropdown(description="Keyword",
                           options=["[ALL]", ]+list(kw_df.kw), value="[ALL]")
        to_conf = Button(button_style="warning",
                         icon="cog", description=f"nn:{sub}")
        output2 = Output()

        to_conf.click = btn_conf_widget(prefix, sub, output2, kw_drop)

        sub_hbox_list.append(to_conf)

        sub_hbox = HBox(sub_hbox_list)
        display(sub_hbox)
        if output is not None:
            display(output)
        display(output2)

    def find_next(prefix=""):
        display(HTML(f"""<h4>
        Prefix:<strong class='text-primary'>"{prefix}"</strong></h4>"""))
        sub_ct = wizard.find_next_level(prefix)

        if sub_ct is None:
            return

        display(HTML(" - ".join(
            list(
                f"{sub} (x{ct})" for sub, ct in zip(sub_ct.index, sub_ct.values))
        )))
        map_dict = dict((f"{sub} ({ct})", (sub, ct))
                        for sub, ct in zip(sub_ct.index, sub_ct.values))

        @interact
        def select_sub(submodule=map_dict):
            sub, ct = submodule
            control_layer(prefix, sub, ct)

    btn_hide = Button(description="hide", button_style="info", icon="folder")
    btn_config_learning = Button(description="setting",
                                 button_style="warning", icon="folder-open")
    controls = HBox(
        [HTML("<h5>Optimizer Details (Optional):</h5>"), btn_hide, btn_config_learning])
    display(controls)
    display(editable)
    over_out = Output(layout={
        "border": "2px dashed #7780FF",
    })

    def start_set_conf():
        over_out.clear_output()
        with over_out:
            find_next("")
            control_layer("", "", 0)

    def end_set_conf():
        over_out.clear_output()

    display(over_out)
    btn_config_learning.click = start_set_conf
    btn_hide.click = end_set_conf
