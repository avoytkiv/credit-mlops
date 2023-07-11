from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
from typing import Dict, Text
from itertools import product
import pandas as pd

class UnsupportedClusterer(Exception):
    def __init__(self, estimator_name):
        self.msg = f"Unsupported estimator {estimator_name}"
        super().__init__(self.msg)


def get_supported_clusterer() -> Dict:
    """
    Returns:
        Dict: supported clusterers
    """
    return {"kmeans": KMeans, "agglo": AgglomerativeClustering, "dbscan": DBSCAN}


def cluster(
    df: pd.DataFrame,
    estimator_name: Text,
    param_grid: Dict
):
    """Cluster data.
    Args:
        df {pandas.DataFrame}: dataset
        estimator_name {Text}: estimator name
        param_grid {Dict}: grid parameters
        cv {int}: cross-validation value
    Returns:
        fitted clusterer
    """
    clusterers = get_supported_clusterer()

    if estimator_name not in clusterers.keys():
        raise UnsupportedClusterer(estimator_name)

    clusterer_class = clusterers[estimator_name]
    X_train = df.values.astype("float32")

    # Get list of all parameter combinations
    param_names = param_grid.keys()
    param_values = param_grid.values()
    all_params = list(product(*param_values))

    best_score = -1  # Initialize best_score to lowest possible silhouette score
    best_model = None

    # Perform grid search manually
    for params in all_params:
        # Create dict of parameters
        param_dict = dict(zip(param_names, params))

        # Initialize and fit model
        model = clusterer_class(**param_dict)
        labels = model.fit_predict(X_train)

        # Compute silhouette score
        score = silhouette_score(X_train, labels)

        # If this model is better than the previous best, update best_score and best_model
        if score > best_score:
            best_score = score
            best_model = model

    # Add an attribute to the model to store the best score
    best_model.best_score_ = best_score

    return best_model
