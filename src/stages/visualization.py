import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger  # noqa: E402

def visualize():
    config = dvc.api.params_show()

    logger = get_logger("VISUALIZE", log_level=config["base"]["log_level"])

    model_name = config["train"]["model"]["model_name"]

    logger.info("Load datasets")
    labeled_df = pd.read_csv(config["data"]["data_labeled"])
    processed_df = pd.read_csv(config["data"]["data_preprocessing"])

    logger.info("Plot clusters")
    logger.info("PLOT1: Create a clusters plot")
    # Create a scatter plot of the featurized data with different colors for each cluster
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=labeled_df.iloc[:, 0], y=labeled_df.iloc[:, 1], hue=labeled_df["cluster"], palette="viridis")
    plt.title(f"Clusters for {model_name}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title='Cluster Label', loc='upper right')
    plt.show()

    plt.savefig(Path(config["reports"]["base_dir"]) / 
                Path(config["reports"]["plots_dir"]) / 
                Path(config["reports"]["plots"]["clusters"]))
    
    logger.info("PLOT2: Create a visualization to compare the clusters")
    # Create a histogram of the clusters across selected features
    processed_df["cluster"] = labeled_df["cluster"]
    # Select columns BALANCE, PURCHASES, CREDIT_LIMIT, PAYMENTS and labels
    selected_columns = ["BALANCE", "PURCHASES", "CREDIT_LIMIT", "PAYMENTS"]
    clusters = processed_df['cluster'].unique()
    # create a pairplot
    # Define the figure size and grid layout properties
    figsize = (14, 10)
    cols = 2  # Number of columns in the grid
    rows = len(selected_columns) // cols  # Number of rows in the grid

    # Create a new figure and a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten the axes array, in case we have only one row (or one column)
    axes = axes.flatten()

    # Loop through the selected columns and create a histogram/density plot for each one
    for i, col in enumerate(selected_columns):
        for cluster in clusters:
            sns.histplot(processed_df[processed_df['cluster'] == cluster][col], 
                        kde=True,  # This enables the density plot
                        ax=axes[i],  # Plot on the i-th subplot
                        label=f'Cluster {cluster}')  # Add a label for the legend
        axes[i].set_title(col)  # Set the title to the column name
        axes[i].legend()  # Show the legend on the i-th subplot

    # Show the plot
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