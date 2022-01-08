__all__ = ["EditableDict", "EditableList",
           "StepByStep"]

from ipywidgets import (
    Button, HTML, Layout, Output, HBox, VBox
)
from typing import List, Dict, Any, Callable
import json
from .html import list_group, list_group_kv
from IPython.display import display


total_width = Layout(width="100%")


class EditableList(VBox):
    """
    Interactive list
    You can add item to the list
    Each added item has a remove button to remove such item
    """

    def __init__(self, data_list: List[Any] = [], pretty_json: bool = True):
        super().__init__([], layout=total_width)
        self.pretty_json = pretty_json
        for data in data_list:
            self+data

    def create_line(self, data: Any) -> None:
        children = list(self.children)
        children.append(self.new_line(data))
        self.children = children
        self.run_update()

    def on_update(self, func: Callable) -> Callable:
        """
        A decorator to set a function
        Every time the list changed
        Will execute this function
        the default arg is the list data
        """
        self.update_func = func
        return func

    def run_update(self):
        if hasattr(self, "update_func"):
            self.update_func(self.get_data())

    def data_to_dom(self, data):
        if self.pretty_json:
            if hasattr(data, "keys"):
                pretty = list_group_kv(data)
            elif type(data) in [list, tuple]:
                pretty = list_group(data)
            else:
                return HTML(f"{data}", layout=total_width)
            return HTML(str(pretty), layout=total_width)
        else:
            return HTML(json.dumps(data))

    def new_line(self, data) -> HBox:
        del_btn = Button(
            description="Nah",
            icon="trash",
            layout=Layout(width="120px"))
        del_btn.button_style = 'danger'
        hbox = HBox([del_btn, self.data_to_dom(data)],
                    layout=total_width, box_style='info')
        hbox.data = data

        def remove_hbox():
            children = list(self.children)
            for i, c in enumerate(children):
                if id(c) == id(hbox):
                    children.remove(c)
            self.children = children
            self.run_update()
        del_btn.click = remove_hbox
        return hbox

    def __add__(self, data: Any):
        self.create_line(data)
        return self

    def get_data(self) -> List[Any]:
        """
        Return the data of this list
        """
        return list(x.data for x in self.children)


class EditableDict(VBox):
    """
    Interactive dictionary
    You can add item to the dictionary
    Each added item has a remove button to remove such item
    """

    def __init__(self, data_dict: Dict[str, Any] = dict(), pretty_json: bool = True):
        super().__init__([], layout=total_width)
        self.pretty_json = pretty_json
        self+data_dict

    def on_update(self, func):
        """
        A decorator to set a function
        Every time the dict changed
        Will execute this function
        the default arg is the dictionary data
        """
        self.update_func = func
        return func

    def run_update(self):
        if hasattr(self, "update_func"):
            self.update_func(self.get_data())

    def create_line(self, key: str, data: Any):
        children_map = dict((child.key, child) for child in self.children)
        children_map[key] = self.new_line(key, data)
        self.children = list(children_map.values())
        self.run_update()

    def data_to_dom(self, data):
        if self.pretty_json:
            if hasattr(data, "keys"):
                pretty = list_group_kv(data)
            elif type(data) in [list, tuple]:
                pretty = list_group(data)
            else:
                return HTML(f"{data}", layout=total_width)
            return HTML(str(pretty), layout=total_width)
        else:
            return HTML(json.dumps(data))

    def new_line(self, key: str, data: Any) -> HBox:
        del_btn = Button(
            description="Nah",
            icon="trash",
            layout=Layout(width="120px"))
        del_btn.button_style = 'danger'
        key_info = HTML(f"<h4 class='text-primary p-1'>{key}</h4>")
        hbox = HBox([VBox([key_info, del_btn]), self.data_to_dom(data)],
                    layout=total_width, box_style='')
        hbox.data = data
        hbox.key = key

        def remove_hbox():
            children = list(self.children)
            for c in children:
                if id(c) == id(hbox):
                    children.remove(c)
            self.children = children
            self.run_update()
        del_btn.click = remove_hbox
        return hbox

    def __setitem__(self, k, v):
        self.create_line(k, v)

    def __add__(self, kv):
        for k, v in kv.items():
            self.create_line(k, v)
        return self

    def get_data(self) -> Dict[str, Any]:
        """
        Return the data of this dict
        """
        return dict((x.key, x.data) for x in self.children)


class LivingStep:
    """
    A step interactive for StepByStep
    """

    def __init__(
        self, func: Callable,
        top_block: HTML = HTML(""),
    ):
        self.output = Output()
        self.func = func
        self.top_block = top_block

    def __call__(self, **kwargs):
        with self.output:
            if self.top_block is not None:
                display(self.top_block)
            return self.func(**kwargs)

    def new_top_block(self, top_block: HTML):
        self.top_block = top_block


class StepByStep:
    """
    A tool to manage progress step by step
    """

    def __init__(
        self,
        funcs: Dict[str, Callable],
        top_board: HTML = None,
        kwargs: Dict[str, Any] = dict()
    ):
        from IPython import get_ipython
        ishell = get_ipython()
        ishell.run_cell_magic("javascript", "", """
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
""")
        self.step_keys: List[str] = list(funcs.keys())
        self.steps: Dict[str, LivingStep] = dict(
            (k, LivingStep(f)) for k, f in funcs.items())
        self.furthest: int = 0
        self.current: int = -1
        self.kwargs: Dict[str, Any] = kwargs
        self.execute_cache: Dict[str, bool] = dict()
        self.top_board: HTML = top_board
        self.page_output: Output = Output()
        self.footer: Output = Output()
        self.create_widget()

    def rerun(self, **kwargs):
        """
        Rerun the current step function
        """
        # find the step
        step: LivingStep = self.steps[self.step_keys[self.current]]
        # clear old output
        step.output.clear_output()
        self.kwargs.update(kwargs)
        step(**self.kwargs)

    def create_control_bar(self,):
        self.bar_hbox = list()
        self.next_btn: Button = Button(
            description="Next", icon='check', button_style='info')
        self.rerun_btn = Button(description="Rerun Step",
                                icon='play', button_style='success')
        self.title = HTML(f"<h4 class='text-primary'>Step By Step</h4>")
        self.next_btn.click = self.next_step
        self.rerun_btn.click = self.rerun
        self.bar_hbox.append(self.title)
        self.bar_hbox.append(self.next_btn)
        self.bar_hbox.append(self.rerun_btn)
        return HBox(self.bar_hbox)

    def create_widget(self) -> None:
        self.vbox_list = []
        if self.top_board is not None:
            self.vbox_list.append(self.top_board)

        # create buttons for progress axis
        self.progress_btns = dict(
            (k, Button(
                description=f"{i+1}:{k}",
                icon="cube",
                button_style="danger"
                if i <= self.furthest else ""))
            for i, (k, v) in enumerate(self.steps.items())
        )
        # assign action to first button
        first_btn: Button = list(self.progress_btns.values())[0]
        first_btn.click = self.to_page_action(0)
        self.progress_bar = HBox(list(self.progress_btns.values()))

        # assemble the entire widget
        self.vbox_list.append(self.progress_bar)
        self.vbox_list.append(self.create_control_bar())
        self.vbox_list.append(self.page_output)
        self.widget = VBox(self.vbox_list)

    def to_page_action(
        self, page_id: int
    ) -> Callable:
        """
        generate the button click function
        """
        def to_page_func():
            return self[page_id]
        return to_page_func

    def update_furthest(self):
        """
        Update the "furthest mark"
        Also enact the next progress button
        """
        if self.furthest < self.current:
            if self.current < len(self):
                # update even button
                btn = self.progress_btns[self.step_keys[self.current]]
                btn.click = self.to_page_action(
                    self.current)
                btn.button_style = 'danger'
            self.furthest = self.current

    def __repr__(self):
        keys = " => ".join(self.step_keys)
        return f"Progress Axis: [{keys}]"

    def __getitem__(self, page_id):
        """
        Display a single page
        """
        if (page_id < 0) or (page_id >= len(self)):
            return
        self.current: int = page_id
        key: str = self.step_keys[page_id]
        step: LivingStep = self.steps[key]
        self.title.value = f"<h4 class='text-danger'>Step {page_id+1}: {key}</h4>"
        self.page_output.clear_output()

        with self.page_output:
            display(step.output)
        if key not in self.execute_cache:
            rt = step(**self.kwargs)
            if hasattr(rt, "keys"):
                self.kwargs(rt)
            self.execute_cache[key] = True

    def next_step(self, **kwargs):
        self.current += 1
        if self.current >= len(self):
            self.current = 0
        self.update_furthest()
        return self[self.current]

    def __len__(self):
        return len(self.step_keys)

    def __call__(self, **kwargs):
        """
        Start the entire progress widget
        """
        display(self.widget)
        display(self.footer)
        self.kwargs.update(kwargs)
        self.next_step(**self.kwargs)

    def show_in(self, step_name: str) -> Callable:
        """
        A decorator that will make the function
            to show under a specific step window
        """
        step = self.steps[step_name]

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                with step.output:
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def show_footer(self, func: Callable):
        """
        A decorator, where functions excuted
            within this, will be showon under footer
        """
        def wrapper(*args, **kwargs):
            with self.footer:
                return func(*args, **kwargs)
        return wrapper
