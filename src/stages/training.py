import argparse
import json
import joblib
import pandas as pd
from pathlib import Path
import sys
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

    model = cluster(
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



if __name__ == "__main__":
    train_model()