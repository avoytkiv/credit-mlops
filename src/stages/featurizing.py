import sys
import argparse
from typing import Text
import yaml
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger


def featurizing(config_path: Text) -> None:
    """
    Load processed data. Apply PCA to reduce the dimensions of the dataset.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("FEATURIZE", log_level=config["base"]["log_level"])
    logger.info("Load processed data")
    df = pd.read_csv(config["data_preprocessing"]["dataset_processed"])
    logger.info("Apply PCA")
    n_components = config["featurize"]["n_components"]
    logger.info("Scale data")
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(df)
    logger.info(f"Number of components: {n_components}")
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_standardized)
    logger.info("Save processed data")
    pd.DataFrame(data_pca).to_csv(config["featurize"]["features_path"], index=False)
    logger.info("Save PCA model")
    joblib.dump(pca, config["featurize"]["model_path"])


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    featurizing(config_path=args.config)