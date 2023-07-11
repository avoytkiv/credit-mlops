import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

def plot_clusters(pca_components, labels):
    """
    Plot clusters in 2D.
    Args:
        pca_components {numpy.ndarray or pandas.DataFrame}: PCA components
        labels {List[int]}: cluster labels

    Example:
        from sklearn.decomposition import PCA

        # Fit your clustering model
        model = cluster(df, estimator_name="kmeans", param_grid={}, cv=5)

        # Get the cluster labels
        labels = model.labels_

        # Compute the PCA components
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(df)

        # Plot the clusters
        plot_clusters(pca_components, labels)
    """
    # Create a scatter plot
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=labels, cmap='viridis')

    plt.show()


def plot_silhouette(df, labels):
    """
    Plot silhouette graph.
    Args:
        df {pandas.DataFrame}: dataset
        labels {List[int]}: cluster labels
    """
    # Compute silhouette score for each sample
    silhouette_values = silhouette_samples(df, labels)

    # Compute overall silhouette score
    silhouette_avg = silhouette_score(df, labels)

    y_lower = 10
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # For each cluster i, plot silhouette scores for each sample in the cluster
    for i in range(len(np.unique(labels))):
        ith_cluster_silhouette_values = silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / len(np.unique(labels)))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()
