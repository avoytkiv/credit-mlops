import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger  # noqa: E402

def visualize():
    config = dvc.api.params_show()
    n_components = config["visualization"]["n_components"]
    random_state = config["base"]["random_state"]
    perplexity = config["visualization"]["perplexity"]
    learning_rate = config["visualization"]["learning_rate"]
    n_iter = config["visualization"]["n_iter"]


    logger = get_logger("VISUALIZE", log_level=config["base"]["log_level"])

    model_name = config["train"]["model"]["model_name"]

    logger.info("Load datasets")
    labeled_df = pd.read_csv(config["data"]["data_labeled"])
    processed_df = pd.read_csv(config["data"]["data_preprocessing"])

    logger.info("Initialize t-SNE")
    tsne = TSNE(n_components=n_components, random_state=random_state, 
                perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    # Fit t-SNE on the labeled data except the labels
    tsne_df = tsne.fit_transform(labeled_df.iloc[:, :-1].values)
    # Add the labels to the t-SNE output
    tsne_df = pd.DataFrame(tsne_df, columns=["Component 1", "Component 2"])
    tsne_df["cluster"] = labeled_df["cluster"]

    logger.info("Plot clusters")
    logger.info("PLOT1: Create a clusters plot")
    # Create a scatter plot of the featurized data with different colors for each cluster
    plt.figure(figsize=(8, 6))
    grid = sns.FacetGrid(tsne_df, hue="cluster", height=8)
    grid.map(plt.scatter, 'Component 1', 'Component 2').add_legend()
    plt.title(f"Clusters for {model_name}")
    plt.show()

    plt.savefig(Path(config["reports"]["base_dir"]) / 
                Path(config["reports"]["plots_dir"]) / 
                Path(config["reports"]["plots"]["clusters"]))
    
    logger.info("PLOT2: Create a visualization to compare the clusters")
    # Create a histogram of the clusters across selected features
    processed_df["cluster"] = labeled_df["cluster"]
    # Select columsn (short version) BALANCE, PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY, CREDIT_LIMIT, MINIMUM_PAYMENTS
    selected_columns = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'CREDIT_LIMIT', 'MINIMUM_PAYMENTS']
    clusters = processed_df['cluster'].unique()

    # mean_df = processed_df.groupby('cluster')[selected_columns].mean()
    # mean_df.to_csv("data/processed/mean_clusters.csv", index=False) 

    # Convert numerical values to categorical labels based on percentiles
    # binned_df = mean_df.copy()
    # num_categories = 3
    # for feature in mean_df.columns:
    #     labels = pd.qcut(mean_df[feature], q=num_categories, labels=['LOW', 'MODERATE', 'HIGH'])
    #     binned_df[feature] = labels
    
    # Define color map for labels
    color_map = {
        'LOW': 'background-color: rgba(255, 0, 0, 0.3)',
        'MODERATE': 'background-color: rgba(255, 255, 0, 0.3)',
        'HIGH': 'background-color: rgba(0, 255, 0, 0.3)'
    }
    # Apply colors to labels in the DataFrame
    # styled_df = binned_df.transpose().style.applymap(lambda label: color_map[label])

    
    # styled_df.to_html('styled_data.html', index=False)
    # binned_df.to_csv("data/processed/binned_clusters.csv", index=False)
    
    num_features = len(selected_columns)
    num_clusters = len(clusters)

    # Create a matrix of subplots
    fig, axes = plt.subplots(nrows=num_features, ncols=num_clusters, figsize=(22, 4*num_features), sharex=False, sharey=False)

    # Loop through each feature and cluster
    for i, feature in enumerate(selected_columns):
        for j, cluster in enumerate(clusters):
            ax = axes[i, j]
            cluster_data = processed_df[processed_df['cluster'] == cluster][feature]
            ax.hist(cluster_data, bins=20, alpha=0.5, label=f'Cluster {cluster}')
            
            # Calculate and annotate the mean value
            mean_value = np.mean(cluster_data)
            ax.axvline(mean_value, color='r', linestyle='dashed', linewidth=1)
            ax.annotate(f'Mean: {mean_value:.2f}', xy=(0.7, 0.8), xycoords='axes fraction', color='r')
            
            if j == 0:
                ax.set_ylabel(feature)
            if i == num_features - 1:
                ax.set_xlabel('Frequency')
            ax.legend()

    plt.tight_layout()
    plt.show()

    plt.savefig(Path(config["reports"]["base_dir"]) / 
                Path(config["reports"]["plots_dir"]) / 
                Path(config["reports"]["plots"]["comparison"]))

    logger.info("PLOT3: Number of samples per cluster")
    # Create a bar plot of the number of samples per cluster
    # Calculate the number of samples in each cluster
    cluster_counts = processed_df['cluster'].value_counts().sort_index()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')

    # Add labels and title
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples per Cluster')
    plt.show()

    plt.savefig(Path(config["reports"]["base_dir"]) / 
                Path(config["reports"]["plots_dir"]) / 
                Path(config["reports"]["plots"]["frequency"]))


if __name__ == "__main__":
    visualize()