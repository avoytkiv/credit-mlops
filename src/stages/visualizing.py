import argparse
import joblib
import pandas as pd
from typing import Text
import yaml
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger  # noqa: E402
from src.utils.visualize_util import plot_clusters # noqa: E402


def visualize(config_path: Text) -> None:
    """Visualize clusters.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("VISUALIZE", log_level=config["base"]["log_level"])

    logger.info("Load model and featurizer")
    model = joblib.load(config["train"]["model_path"])
    pca = joblib.load(config["featurize"]["model_path"])
    # Labels are the cluster numbers
    labels = model.labels_
    
    logger.info("Featurize data")
    df = pd.read_csv(config["data_preprocessing"]["dataset_processed"])
    pca_components = pca.transform(StandardScaler().fit_transform(df))
    
    logger.info("Plot clusters")
    plot_clusters(pca_components, labels)

    logger.info("Save plot")
    plt.savefig(Path(config["reports"]["plots_dir"]) / "clusters.png")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    visualize(config_path=args.config)