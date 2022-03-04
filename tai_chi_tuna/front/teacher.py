
"""
Teacher is the module that can assign html panel to any class
"""

from pathlib import Path
from typing import Type, Any
from PIL import Image
from io import BytesIO
import base64
import regex
from .html import DOM
import tai_chi_tuna


def image_to_base64(img: Image) -> str:
    """
    Transform PIL Image to base64 for frontend display
    """
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return f"data:image/jpeg;base64,{base64_str.decode()}"

class Teacher:
    union = dict() # records holding {class name: teacher instances}
    rendered = dict() # cache holding {class name: rendered html}
    """
    The decorator to any class, to assign visualizing teacher
    @Teacher("explain_to_some_class.html")
    class SomeClass:
        def __init__(self)...
    """
    
    def __init__(self, html_path: Path):
        self.html_path = html_path
        if hasattr(self, "static_folder")==False:
            raise ImportError(f"Please use TunaTeacher or EngineTeacher instead")

    def __call__(self, cls:Type) -> Type:
        """
        This step is executed when teacher is used as decorator upon class
        """
        self.teacher_static = self.static_folder/"teacher"
        self.teacher_html = self.teacher_static/self.html_path
        cls.teacher = self
        self.name = cls.__name__
        self.union[cls.__name__] = self
        return cls

    def __repr__(self):
        name = self.name if hasattr(self, "name") else "Unassigned"
        return f"(Teacher@{name})"

    def find_image_and_replace(self, text, ):
        """
        Find the position of '{{xxx}}' in text
        and return it's data form
        """
        for found in regex.findall(r"{{[^}]+}}", text):
            with Image.open(self.static_folder/"img"/found[2:-2]) as img:
                text = text.replace(found, image_to_base64(img))
        return text

    def read(self):
        if self.name in self.rendered:
            # read from cache
            return self.rendered[self.name]
        try:
            with open(self.teacher_html, "r") as f:
                result = self.find_image_and_replace(f.read())
                self.rendered[self.name] = result
                return result
        except FileNotFoundError:
            """
            If education failed, let's not stand
            in the way of actually use the software
            """
            pass

class TunaTeacher(Teacher):
    static_folder = Path(tai_chi_tuna.__file__).parent/"static"

def teach(obj: Any, execute:bool = True):
    """
    Teach an object to use the teacher

    # pass object as following to display the content of the object
    teach(some_object)
    """
    if hasattr(obj, "teacher"):
        rendered = obj.teacher.read()
        if rendered is None:
            return 
        dom = DOM(rendered,"span")
        if execute:
            dom()
        else:
            return dom
    else:
        return




