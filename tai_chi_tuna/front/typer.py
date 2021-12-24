__all__ = ["InteractiveTyping", "INT", "BOOL", "FLOAT",
           "STR", "LIST"]
from ipywidgets import (
    VBox, HBox, HTML, Layout, Button, Output,
    Text, Textarea, IntSlider, FloatSlider, SelectMultiple, Dropdown, Checkbox
)
from typing import List, Any


class InteractiveTyping:
    """
    Typing for interactive details
    self.__call__() will create widgets directly
    """
    name = "anything"
    is_typing = True

    def solid(self, default) -> None:
        """
        Reset default value
        """
        if default is not None:
            self.default = default


class INT(InteractiveTyping):
    def __init__(self, min_: int = 0, max_: int = 10, step: int = 1, default: int = None):
        self.max_ = max_
        self.min_ = min_
        self.step = step
        self.default = default if default is not None else 1

    def __repr__(self):
        return f"int[{self.min_}-{self.max_}, :{self.step}]={self.default}"

    def __call__(self, default: int = None):
        self.solid(default)
        return IntSlider(
            value=self.default,
            min=self.min_,
            max=self.max_,
            step=self.step,
        )


class BOOL(InteractiveTyping):
    def __init__(self, name: str = "", default: bool = True,):
        self.default = default
        self.name = name

    def __repr__(self):
        return f"bool={self.default}"

    def __call__(self, default: bool = None) -> Checkbox:
        self.solid(default)
        return Checkbox(value=self.default, description=self.name)


class FLOAT(InteractiveTyping):
    def __init__(self, min_: int = -1., max_: int = 1., step: int = .01, default: int = None):
        self.max_ = max_
        self.min_ = min_
        self.step = step
        self.default = default if default is not None else 0.01

    def __repr__(self):
        return f"float[{self.min_}-{self.max_}, :{self.step}]={self.default}"

    def __call__(self, default: int = None):
        self.solid(default)
        return FloatSlider(
            value=self.default,
            min=self.min_,
            max=self.max_,
            step=self.step,
        )


class STR(InteractiveTyping):
    """
    String object
    will create text or textarea
    """

    def __init__(self, default: str = None, use_area: bool = False):
        """
        use_area: do we use Textarea, if False,we use Text
        """
        self.default = "" if default is None else default
        self.use_area = use_area

    def __repr__(self):
        return f"str='{self.default}'"

    def __call__(self, default: str = None):
        self.solid(default)
        if self.use_area:
            return Textarea(value=self.default, layout=Layout(width="80%"))
        return Text(value=self.default)


class LIST(InteractiveTyping):
    """
    dropdown list type or multiselection type
    """

    def __init__(self, options: List[Any] = [], default: Any = None, multi: bool = False):
        """
        if multi: default should be iterable
        else: default should be one of the option
        """
        self.options = options
        self.default = default
        self.multi = multi

    def __repr__(self):
        if self.multi:
            size = f"[0-{self.default}]/{len(self.options)}"
        else:
            size = f"1/{len(self.options)}"
        return f"list,{size}"

    def __call__(self, default: Any = None):
        self.solid(default)
        if self.multi:
            inter = SelectMultiple(options=self.options)
        else:
            inter = Dropdown(options=self.options)

        if self.default is not None:
            # if multi: default should be iterable
            # else: default should be one of the option
            inter.value = self.default
        return inter
