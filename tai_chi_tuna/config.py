from pathlib import Path
from typing import Dict
import json
import logging


class PhaseConfig:
    """
    A configuration management mechanism
    """
    is_phase = True

    def __init__(self, **kwargs):
        """
        Please initiate the class
            with classmethod load(PROJECT_NAME)
        """
        self.config = dict()
        self.config.update(kwargs)

    def __setitem__(self, k, v):
        self.config[k] = v

    def __getitem__(self, k):
        return self.config[k]

    def __delitem__(self, k):
        del self.config[k]

    def __contains__(self, k):
        return k in self.config

    def __call__(self):
        return self.get_data(self.config)

    def get_data(self, raw):
        """
        Reconstruct back to dict or list or value format
        """
        if hasattr(raw, "is_phase"):
            return raw.get_data(raw.config)
        if type(raw) == list:
            raw = list(self.get_data(i) for i in raw)
            return raw
        if type(raw) == dict:
            for k, v in raw.items():
                raw[k] = self.get_data(v)
            return raw
        return raw

    def __str__(self):
        return json.dumps(self(), indent=2)

    def __repr__(self,):
        return f"PhaseConfig:{self}"

    @classmethod
    def load(
        cls,
        project: Path = Path("./.tai-chi"),
        new: bool = False,
    ):
        """
        Load phase config from a project folder
        """
        project = Path(project)
        project.mkdir(exist_ok=True, parents=True)
        config_path = project/"phase.json"
        # load existed phase config
        if config_path.exists() and (new == False):
            logging.warning(
                f"Loading phase config from {config_path}")
            with open(config_path, "r") as f:
                data = json.loads(f.read())
            obj = cls(**data)
            logging.info(
                f"We found existence following keys: {list(data.keys())}")
        # create new phase config
        else:
            obj = cls()
        obj.project = project
        return obj

    def save(self, project: Path = None):
        # make sure of project
        if project is None:
            if hasattr(self, "project") == False:
                self.project = Path("./.tai-chi")
        else:
            self.project = Path(project)

        self.project.mkdir(exist_ok=True, parents=True)
        config_path = self.project/"phase.json"
        # write json to config file
        with open(config_path, "w") as f:
            f.write(str(self))
