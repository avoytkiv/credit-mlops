import json
import joblib
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger  # noqa: E402
from src.utils.train_util import cluster  # noqa: E402


def train_model() -> None:
    """Train model."""

    config = dvc.api.params_show()

    logger = get_logger("TRAIN", log_level=config["base"]["log_level"])

    logger.info("Get model name")
    model_name = config["train"]["model"]["model_name"]
    logger.info(f"Model: {model_name}")

    logger.info("Load featurized dataset")
    train_df = pd.read_csv(config["data"]["data_featurized"])

    logger.info("Train model")

    model, labels = cluster(
        df=train_df,
        model_name=model_name,
        param_grid=config["train"]["model"]["param_grid"]
    )
    logger.info(f"Best score: {model.best_score_}")

    logger.info("Save model")
    models_path = config["train"]["model_path"]
    joblib.dump(model, models_path)

    logger.info("Report training results")
    report = {"silhouette_score": str(model.best_score_)}

    logger.info("Save metrics")
    metrics_path = Path(config["reports"]["base_dir"]) / config["reports"]["metrics_file"]
    json.dump(report, open(metrics_path, "w"))
    logger.info("Metrics saved")

    # Append cluster labels to featurized dataset and save
    train_df["cluster"] = labels
    train_df.to_csv(Path(config["data"]["data_labeled"]), index=False)

    logger.info("Plot clusters")
    # Create a scatter plot of the featurized data with different colors for each cluster
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=train_df.iloc[:, 0], y=train_df.iloc[:, 1], hue=labels, palette='viridis')
    plt.title(f"Clusters for {model_name}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title='Cluster Label', loc='upper right')
    plt.show()

    logger.info("Save plot")
    plt.savefig(Path(config["reports"]["base_dir"]) / 
                Path(config["reports"]["plots_dir"]) / 
                Path(config["reports"]["plots"]["clusters"]))



if __name__ == "__main__":
    train_model()