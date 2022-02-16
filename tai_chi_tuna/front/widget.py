__all__ = ["InteractiveAnnotations", "interact_intercept", "init_interact",
           "reconfig_manual_interact"]

from typing import Callable, Dict, Any
from ipywidgets import (
    interact_manual, Button, VBox, Output
)
from .html import Flash
import json
from IPython.display import display


class InteractiveAnnotations:
    """
    Build interactive based on
        the info of function's ```__annotations__```
    """

    def __init__(
        self, func: Callable,
        icon: str = "rocket",
        description: str = 'Run',
        button_style='primary'
    ):
        self.func = func
        self.icon = icon
        self.button_style = button_style
        self.description = description
        self.build_vbox(func)
        self.out = Output()
        display(self.out)

    @classmethod
    def on(
        cls,
        callback: Callable,
        icon: str = 'rocket',
        description: str = 'Run',
        button_style: str = 'primary'
    ) -> Callable:
        """
        Use this class as a decorator
        @InteractiveAnnotation.on(callback)
        def target_func(a:STR(), b:INT()=1):
            ...
        """
        def decorator(func: Callable):
            obj = cls(
                func,
                icon=icon,
                description=description,
                button_style=button_style
            )
            display(obj.vbox)
            obj.register_callback(callback=callback)
            return func
        return decorator

    def build_vbox(self, func: Callable):
        row_list = []
        self.fields = dict()
        for k, v in func.__annotations__.items():
            if hasattr(v, "is_typing") == False:
                continue
            widget = v()
            widget.description = k
            row_list.append(widget)
            self.fields.update({k: widget})

        # final button
        self.final_btn = Button(
            description=self.description,
            icon=self.icon,
        )
        self.final_btn.button_style = self.button_style
        row_list.append(self.final_btn)

        # create interactive
        self.vbox = VBox(row_list)
        return self.vbox

    def register_callback(
        self,
        callback: Callable
    ) -> None:
        def run_callback():
            with self.out:
                kwargs = self()
                self.latest_data = kwargs
                callback(kwargs)
        self.final_btn.click = run_callback

    def __call__(self) -> Dict[str, Any]:
        """
        extract interactive data values
        """
        rt = dict()
        for k, widget in self.fields.items():
            rt.update({k: widget.get_interact_value()})
        return rt


def reconfig_manual_interact(
    widget,
    description: str = "Create",
    button_style: str = "primary",
    icon: str = "plus"
) -> Button:
    """
    reconfigure the button of interactive features
    """
    btn = None
    for w in widget.children:
        if type(w) == Button:
            btn = w
            break
    btn.description = description
    btn.button_style = button_style
    btn.icon = icon
    return btn


def print_kwargs(kwargs) -> Dict[str, Any]:
    Flash.info(json.dumps(kwargs, indent=2))
    return kwargs


def interact_intercept(
    func: Callable,
    result_cb: Callable = print_kwargs
):
    """
    Initialize a class with interactive features
    func: the original decorated function for interact_manual,
        the annotation of the function will be interpreted
    result_cb: the callback function to process the result
    """
    annotations = func.__annotations__
    defaults = func.__defaults__
    kwargs = dict()
    if defaults is not None:
        for (k, typing), default in zip(annotations.items(), defaults):
            kwargs.update({k: typing(default)})
    obj = dict()

    def fillin_init(**kwargs):
        obj.update({
            "kwargs": kwargs,
        })
    f = interact_manual(fillin_init, **kwargs)

    btn = reconfig_manual_interact(f.widget)
    out = Output()
    display(out)
    if btn is not None:
        original = btn.click

        def new_click_event():
            """
            This function will be called
            When the interact_manual's button is clicked
            """
            with out:
                original()
                res = result_cb(obj['kwargs'])
            return res
        btn.click = new_click_event

    return obj, f


def init_interact(cls, result_cb: Callable = print_kwargs):
    return interact_intercept(cls.__init__, result_cb=result_cb)
