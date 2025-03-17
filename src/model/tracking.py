import pandas as pd
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path

from ..utils.file_io import save_file

class Stage(Enum):
    TEST = auto()
    VAL = auto()

class Experiment:
    """Handles the logging and tracking of machine learning experiments, including metrics, models, 
       and artifacts for different run stages."""
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.metrics = defaultdict(list)
        self.models = {}
        self.artifacts = {}
        self.split_metrics = []
        self.stage = None
        self.model_info = None
        
    def start_run(self, info : dict, stage):
        self.model_info  = info
        self.stage = stage
        self.split_metrics = []

    def add_split_metrics(self, values):
        self.split_metrics.append(values)

    def add_metrics(self, metrics):
        self.metrics[self.stage.name].append(metrics)

    def end_run(self):
        run_metrics = pd.DataFrame(self.split_metrics).mean().to_dict()
        self.add_metrics(dict(self.model_info, **run_metrics))
        self.split_metrics = []
        self.model_info = None

    def add_artifact(self, artifacts):
        self.artifacts = dict(self.artifacts, **artifacts)

    def add_model(self, model_name, model):
        self.models[model_name] = model


    def log(self):

        # log metrics
        save_file(self.metrics, self.log_path / 'metrics.json')

        # log artifacts
        for name, artifact in self.artifacts.items():
            extension = '.json' if isinstance(artifact, dict) else '.png'
            save_file(artifact, self.log_path / (str(name) + extension))

        # log models
        for name, model in self.models.items():
            save_file(model, self.log_path / (str(name) + '.joblib'))   
