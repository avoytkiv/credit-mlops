import argparse
import json
import joblib
import pandas as pd
from typing import Text
import yaml
from pathlib import Path
import sys

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger  # noqa: E402
from src.utils.train_util import cluster  # noqa: E402


def train_model(config_path: Text) -> None:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("TRAIN", log_level=config["base"]["log_level"])

    logger.info("Get estimator name")
    estimator_name = config["train"]["estimator_name"]
    logger.info(f"Estimator: {estimator_name}")

    logger.info("Load featurized dataset")
    train_df = pd.read_csv(config["featurize"]["features_path"])

    logger.info("Train model")

    model = cluster(
        df=train_df,
        estimator_name=estimator_name,
        param_grid=config["train"]["estimators"][estimator_name]["param_grid"]
    )
    logger.info(f"Best score: {model.best_score_}")

    logger.info("Save model")
    models_path = config["train"]["model_path"]
    joblib.dump(model, models_path)

    logger.info("Report training results")
    report = {"silhouette_score": str(model.best_score_)}

    logger.info("Save metrics")
    metrics_path = Path(config["reports"]["reports_dir"]) / config["reports"]["metrics_file"]
    json.dump(report, open(metrics_path, "w"))
    logger.info("Metrics saved")



if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)