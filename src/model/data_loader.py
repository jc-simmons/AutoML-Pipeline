from pathlib import Path
import numpy as np

from ..utils.file_io import load_file
from ..model.preprocessing import create_preprocessor

class Dataset:
    """ Handles loading and preprocessing of dataset. """
    def __init__(self, config):
        self.path = Path(config['paths']['data']) / config['files']['cleaned_data']
        self.features = config['features']
        self.target = config['target']
        self.X, self.y = self._load_and_preprocess()

    def _load_and_preprocess(self):
        data = load_file(self.path)
        # feature selection
        all_features = [feature for ftype in self.features.values() for feature in ftype]
        select_data = data[all_features+self.target]
        # preprocessing
        preprocessor = create_preprocessor()
        filtered_data = preprocessor.fit_transform(select_data)
        X, y = filtered_data[all_features], filtered_data[self.target]
        return X, y
    
  
class DataLoader:
    """ Handles split indices based on input stages. """
    def __init__(self, dataset):
        self.dataset = dataset
        self.stages = {}

    def __call__(self, stage):
        return self.__iter__(stage)
    
    def __getitem__(self, index):
        X, y = self.dataset.X.iloc[[index]], self.dataset.y.iloc[[index]]
        return X, y

    def __iter__(self, stage=None):
        
        splits = self.stages[stage]     
        for train_indices, test_indices in splits:

            X_train, X_test = self.dataset.X.iloc[train_indices], self.dataset.X.iloc[test_indices]
            y_train, y_test = self.dataset.y.iloc[train_indices], self.dataset.y.iloc[test_indices]

            yield X_train, X_test, y_train, y_test


def train_test_val_indices(X, y, test_splitter, val_splitter):
    """ Computes train, validation, test indices based on split generators. """
    Xp = np.array(X)
    yp = np.array(y)
    train_inds, test_inds = next(test_splitter.split(Xp, yp))
    train_test_indices = [tuple([train_inds, test_inds])] 
    
    validation_splits = val_splitter.split(Xp[train_inds], yp[train_inds])
    validation_indices = []

    # get the validation indices relative to the pre-split data
    for val_train, val_test in validation_splits:
       validation_indices.append((train_inds[val_train], train_inds[val_test]))
        
    return train_test_indices, validation_indices
