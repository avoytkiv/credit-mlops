from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
from typing import Dict, Text
from itertools import product
import pandas as pd
import dvc.api

class UnsupportedClusterer(Exception):
    def __init__(self, model_name):
        self.msg = f"Unsupported estimator {model_name}"
        super().__init__(self.msg)


def get_supported_clusterer() -> Dict:
    """
    Returns:
        Dict: supported clusterers
    """
    return {"kmeans": KMeans, 
            "agglo": AgglomerativeClustering, 
            "dbscan": DBSCAN, 
            "hdbscan": HDBSCAN}


def cluster(
    df: pd.DataFrame,
    model_name: Text,
    param_grid: Dict
):
    """Cluster data.
    Args:
        df {pandas.DataFrame}: dataset
        model_name {Text}: estimator name
        param_grid {Dict}: grid parameters
        cv {int}: cross-validation value
    Returns:
        fitted clusterer
    """
    config = dvc.api.params_show()

    clusterers = get_supported_clusterer()

    if model_name not in clusterers.keys():
        raise UnsupportedClusterer(model_name)

    clusterer_class = clusterers[model_name]
    X_train = df.values.astype("double")

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
        # Set random seed for KMeans
        seed = config["base"]["random_state"]
        if model_name == "kmeans":
            param_dict['random_state'] = seed
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

    return best_model, labels
