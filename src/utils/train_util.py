from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
from typing import Dict, Text
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
    param_grid: Dict,
    cv: int,
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

    clusterer = clusterers[estimator_name]()
    silhouette_scorer = make_scorer(silhouette_score)
    clf = GridSearchCV(
        estimator=clusterer, param_grid=param_grid, cv=cv, verbose=1, scoring=silhouette_scorer
    )
    X_train = df.values.astype("float32")
    clf.fit(X_train)

    return clf
