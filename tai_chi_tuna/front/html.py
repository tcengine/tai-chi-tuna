__all__ = ["DOM", "list_group", "list_group_kv",
           "col_sm", "img_dom", "data_url", "image_to_base64",
           "Flash"]

import math
from IPython.display import HTML as DISPLAY_HTML
from io import BytesIO
import logging
import base64
from PIL.Image import Image as ImageClass
from typing import Any, List, Dict
from ipywidgets import (
    HTML,
    Layout,
    HBox,
    Output,
    Button
)
from IPython.display import display

HAVE_DISPLAY = True


class DOM:
    """
    A helper function to generate DOM in HTML
    """

    def __init__(
        self, txt: str,
        tag: str,
        attrs: Dict[str, str] = dict()
    ):
        self.txt = txt
        self.tag = str(tag).lower()
        self.attrs = attrs
        self.refresh_attr()

    @staticmethod
    def extend(text, tag, **kwargs):
        attributes = (" ".join(f'{k}="{v}"' for k, v in kwargs.items()))
        attributes = " "+attributes if attributes else attributes
        start = f"<{tag}{attributes}>"
        inner = f"{text}"
        end = f"</{tag}>"
        text = f"{start}{inner}{end}"
        return start, inner, end

    def refresh_attr(self):
        self.start, self.inner, self.end = self.extend(
            self.txt, self.tag, **self.attrs)

    def __mul__(self, new_tag):
        assert type(new_tag) == str
        return DOM(self.text, new_tag)

    def __add__(self, dom):
        return self.text+dom.text

    def __repr__(self) -> str:
        return f"{self.start}{self.inner}{self.end}"

    def __getitem__(self, k):
        return self.attrs[k]

    def __setitem__(self, k, v):
        self.update({k, v})

    def __call__(self):
        self.display()

    @property
    def text(self) -> str:
        return str(self)

    def append(self, subdom):
        self.inner = self.inner+str(subdom)
        return self

    def update(self, dict_):
        self.attrs.update(dict_)
        self.refresh_attr()
        return self

    def display(self):
        display(DISPLAY_HTML(self.text))


def image_to_base64(
    img: ImageClass
) -> str:
    """
    Transform PIL Image to base64 for API
    Return:
        - base64 encoded image bytes
    """
    img = img.convert('RGB')
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()


def data_url(
    img: ImageClass
) -> str:
    """
    Return:
        - data url string,
            can be used as the src value of <img>
    """
    return f"data:image/jpg;base64,{image_to_base64(img)}"


def img_dom(img: ImageClass) -> DOM:
    return DOM("", "img", {"src": data_url(img)})


def deeper(x):
    if type(x) in [list, set, tuple]:
        return list_group(x)
    if type(x) == dict:
        return list_group_kv(x)
    if type(x) in [int, str, float, bool]:
        return x
    if type(x) == ImageClass:
        return img_dom(x)
    return x


def col_sm(iterable: List[Any], portions: List[float] = None,):
    """
    Create a DOM that will create a div with col-sm-<portion>
    len(portions) must be equal to len(iterable)
    """
    if portions == None:
        portions = [math.floor(12/len(iterable)), ] * len(iterable)
    row = DOM("", "div", {"class": "row"})
    for i, p in zip(iterable, portions):
        row.append(DOM(i, "div", {"class": f"col-sm-{p}"}))
    return row


def list_group(iterable: List[Any]) -> DOM:
    """
    Create a DOM that will display a bootstrap list group
    """
    table = DOM("", "table", {"class": ".table"})
    for i in iterable:
        row = DOM("", "tr", {"class":"tr", "style": "color:#FF5522;padding:2px"})
        cell = DOM(
            deeper(i), "td",
            {"class":"td",
            "style": "color:#333355;padding:2px;border-bottom:1px solid #333355"})
        row.append(cell)
        table.append(row)
    return table


def list_group_kv(data: Dict[str, Any]) -> DOM:
    """
    Create a DOM, using a dictionary
        that will display a bootstrap list group
    """
    result = DOM("", "table", {"class": ".table"})
    for k, v in data.items():
        row = DOM("", "tr", {"class": "tr"})
        row.append(DOM(f"{k}:", "th", {"class": "font-weight-bold th", "style":"color:#FF5522;padding:2px"}))\
            .append(DOM(
                f"{deeper(v)} ",
                "td",
                {"class": "font-weight-bold td",
                "style":"color:#333355;padding:2px;border-bottom:1px solid #333355"}))
        result.append(row)
    return result


class Flash:
    """
    Show alert message in the frontend output
    Flash.warning("Something will be wrong", key="Warn!")
    """
    @staticmethod
    def create_msg_box(color, text, key: str = None):
        text = str(text)
        if key is not None:
            key = f"<strong>{key}</strong> "
        else:
            key = ""
        text_bar = HTML(f"""<div class='alert alert-{color}' role='alert'>
        {key} {text}</div>""", layout=Layout(width='95%'))
        close_btn = Button(description="x", layout=Layout(width='3%'))

        total = HBox([text_bar, close_btn])

        def close_bar():
            total.close()
        close_btn.click = close_bar
        return total

    @classmethod
    def get_info(cls, text, key: str = None):
        return cls.create_msg_box('info', text, key)

    @classmethod
    def get_warning(cls, text, key: str = None):
        return cls.create_msg_box('warning', text, key)

    @classmethod
    def get_danger(cls, text, key: str = None):
        return cls.create_msg_box('danger', text, key)

    @classmethod
    def get_success(cls, text, key: str = None):
        return cls.create_msg_box('success', text, key)

    @classmethod
    def info(cls, text, key: str = None):
        if HAVE_DISPLAY:
            display(cls.get_info(text, key))
        else:
            logging.info(f"{key}:{text}")

    @classmethod
    def warning(cls, text, key: str = None):
        if HAVE_DISPLAY:
            display(cls.get_warning(text, key))
        else:
            logging.warning(f"{key}:{text}")

    @classmethod
    def danger(cls, text, key: str = None):
        if HAVE_DISPLAY:
            display(cls.get_danger(text, key))
        else:
            logging.error(f"{key}:{text}")

    @classmethod
    def success(cls, text, key: str = None):
        if HAVE_DISPLAY:
            display(cls.get_success(text, key))
        else:
            logging.debug(f"{key}:{text}")
