import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class DropNA(BaseEstimator, TransformerMixin):
    """Transforms input by dropping any rows containing missing values (NaN)."""
    def __init__(self): 
        return 
    def fit(self, X, y=None):
        return self 
    def transform(self, X):
        X = X.dropna()
        return X
        

class FlattenTransformer(BaseEstimator, TransformerMixin):
    """ Transforms input by flattening it (removing any single-dimensional axes). """
    def __init__(self): 
        return 
    def fit(self, X, y=None):
        return self 
    def transform(self, X):
        X = X.squeeze()
        return X


def create_preprocessor(features = None):
    """ Constructs preprocessor for prepping data prior to model transforms. """
    preprocessor = Pipeline([
        ('drop NaN', DropNA())
    ])
    return preprocessor


def create_feature_transformer(features):
    """ Constructs a task-specific target transformer for different feature types. """
    transformer = ColumnTransformer([
    ("scaling", MinMaxScaler(), features['numeric']),
    ("encoding", OneHotEncoder(), features['categorical'])
    ])
    return transformer


def create_target_transformer(target = None):
    """ Constructs task-specific transformer to handle any pre or post-prediction modifications to target. """
    transformer = Pipeline([
     ('squeeze', FlattenTransformer())
    ])
    return transformer