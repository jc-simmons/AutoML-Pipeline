import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score


def compute_metrics(estimator, X, y):
    """ Task-specific metrics to return. """
    y, y_pred = np.asarray(y), np.asarray(estimator.predict(X))  
    y_bins, y_pred_bins = percentile_bins(y, y_pred, percentiles = [0.25, 0.5, 0.75])

    metrics = {
        'R2': r2_score(y, y_pred),
        'ACC': accuracy_score(y_bins, y_pred_bins)
    }

    return metrics

def compute_artifacts(estimator, X, y):
    """ Task-specific figures/artifacts to return. """
    y_pred = estimator.predict(X)
    y_bins, y_pred_bins = percentile_bins(y, y_pred, percentiles = [0.25, 0.5, 0.75])

    artifacts = {
        'feature_importance': aggregate_feature_importance(X, estimator.predictor, 
                                                           estimator.feature_transformer),
        'confusion_matrix': create_confusion_matrix(y_bins, y_pred_bins)
    }

    return artifacts


def create_confusion_matrix(y, y_pred, class_labels=None):
    """ Creates the confusion matrix plot, returns figure. """

    conf_matrix = confusion_matrix(y, y_pred)
    
    # If class labels are not provided, infer them from the target variable.
    if class_labels is None:
        class_labels = sorted(set(y.flatten())) 
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sn.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", ax=ax,
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    return fig


def aggregate_feature_importance(X, model, feature_transformer = None):
    """ Extracts feature importances and aggregates them if the feature has been split into multiple encodings. """
    feature_names = X.columns.tolist()
    transformed_feature_importances = model.feature_importances_
    feature_importance = {}

    if feature_transformer:
        transformed_feature_names = feature_transformer.get_feature_names_out()
    else:
        transformed_feature_names = feature_names

    for feature in feature_names:
        aggregated_importance = 0
        for index, transformed_feature in enumerate(transformed_feature_names):
            if feature in transformed_feature:
                aggregated_importance += transformed_feature_importances[index]

        feature_importance[feature] = aggregated_importance

    # sort the insertion order so that the values display in descending order
    sorted_importance= {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}

    return sorted_importance


def percentile_bins(y, y_pred, percentiles):
    """ Places input y and y_pred into percentile bins computed with y. """
    thresholds =  list(percentiles) 
    thresholds = np.percentile(y, np.array(thresholds) * 100) 
    y_bins = np.digitize(y, thresholds, right=True) 
    y_pred_bins = np.digitize(y_pred, thresholds, right=True) 
    return y_bins, y_pred_bins