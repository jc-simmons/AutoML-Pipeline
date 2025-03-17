import random

import numpy as np

from ..utils.file_io import load_file
from ..utils.config_helpers import dyanmic_loader, generate_model_combinations
from ..model.data_loader import Dataset, DataLoader, train_test_val_indices
from ..model.preprocessing import create_feature_transformer, create_target_transformer
from ..model.tracking import Experiment, Stage
from ..model.model_runner import ModelRunner


def execute():

    config = load_file('config.yaml')
    RANDOM_SEED = config['random_seed']
    LOG_PATH = config['paths']['output']
    REFIT = config['refit']
    MDL_CONFIG = config['models']
    TST_CONFIG = next(iter(config['test_split'].items()))
    VAL_CONFIG = next(iter(config['validation_split'].items()))

    set_random_state(RANDOM_SEED)

    # Setup the experiment tracking and evaluation
    experiment = Experiment(log_path=LOG_PATH)

    # Load data and processing methods
    dataset = Dataset(config)

    feature_transformer = create_feature_transformer(dataset.features)
    target_transformer = create_target_transformer(dataset.target)
    
    # Create generate the train-test indices and the indices of the validation splits
    train_test_splitter = dyanmic_loader(TST_CONFIG)
    validation_splitter = dyanmic_loader(VAL_CONFIG)

    train_test_indices, validation_indices = train_test_val_indices(
        dataset.X, dataset.y, train_test_splitter, validation_splitter)
    
    # Create a dataloader and set iteration splits based on stage
    dataloader = DataLoader(dataset)
    dataloader.stages[Stage.VAL] = validation_indices
    dataloader.stages[Stage.TEST] = train_test_indices

    # import and initialize models
    model_loader = generate_model_combinations(MDL_CONFIG)

    model_runner = ModelRunner(experiment, dataloader, feature_transformer, target_transformer)

    print("Starting model evaluation..")
    model_runner.stage = Stage.VAL
    for model, model_info in model_loader:
        model_runner.run(model, model_info)
        print(f'Done model: {model_info}')

    if REFIT:
        # Get the best model and do final train/test run
        best_model_id = max(range(len(experiment.metrics[Stage.VAL.name])), key=lambda i: experiment.metrics[Stage.VAL.name][i]['R2'])
        model_runner.stage = Stage.TEST
        final_model, model_info = model_loader[best_model_id]
        model_runner.run(final_model, model_info)


    experiment.log()


def set_random_state(seed):
    """ Sets the global random state. """
    if not isinstance(seed, int):
        seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    execute()
