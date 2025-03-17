class Predictor:
    """ Encapsulates a model with optional feature and target transformers. """
    def __init__(self, predictor, feature_transformer = None, target_transformer = None):
        self.predictor = predictor
        self.feature_transformer = feature_transformer
        self.target_transformer = target_transformer


    def fit(self, X, y):        
        if self.feature_transformer:
            X = self.feature_transformer.fit_transform(X, y)
        if self.target_transformer:
            y = self.target_transformer.fit_transform(y)

        self.predictor.fit(X, y)

    def predict(self, X):       
        if self.feature_transformer:
            X = self.feature_transformer.transform(X)
        y = self.predictor.predict(X)
        if hasattr(self.target_transformer, 'inverse_transform'):
            y = self.target_transformer.inverse_transform(y)

        return y

    def predict_proba(self, X):  
        if self.feature_transformer:
            X = self.feature_transformer.transform(X) 

        if hasattr(self.predictor, "predict_proba"):
            return self.predictor.predict_proba(X)
