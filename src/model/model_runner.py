from src.model.tracking import Stage
from src.model.evaluation import compute_metrics, compute_artifacts
from src.model.predictor import Predictor


class ModelRunner:
    """ Orchestrates model execution and evaluation based on experiment stage. """
    def __init__(self, experiment, data_loader, feature_transformer = None, target_transformer = None):
        self.stage = None
        self.experiment = experiment
        self.data_loader = data_loader
        self.feature_transformer = feature_transformer
        self.target_transformer = target_transformer

    def run(self, model, model_info):

        self.experiment.start_run(model_info, self.stage)
        estimator = Predictor(
            model, 
            self.feature_transformer,
            self.target_transformer
        )
        
        for X_train, X_test, y_train, y_test in self.data_loader(self.stage):   
            estimator.fit(X_train, y_train)
            self.run_eval(estimator, X_test, y_test)
        self.experiment.end_run()

    def run_eval(self, estimator, X, y):

        metrics = compute_metrics(estimator, X, y)
        self.experiment.add_split_metrics(metrics)

        if self.stage == Stage.TEST:
            artifacts = compute_artifacts(estimator, X, y)
            self.experiment.add_artifact(artifacts)
            self.experiment.add_model('model', estimator)
