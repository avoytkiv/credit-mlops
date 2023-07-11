import sys
import argparse
from typing import Text
import yaml
from pathlib import Path
import pandas as pd

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from src.utils.logs import get_logger


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:  # Only for numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            filter = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
            df_no_outliers = df.loc[filter]

    return df_no_outliers


def data_load(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("DATA_LOAD", log_level=config["base"]["log_level"])

    logger.info("Get dataset")
    df = pd.read_csv(config["data_load"]["dataset_csv"])
    logger.info("Clean dataset")
    logger.info("Drop missing values")
    df = df.dropna()
    logger.info("Drop CUST_ID column")
    df = df.drop('CUST_ID', axis=1)
    logger.info("Remove outliers")
    df = remove_outliers(df)
    logger.info("Save raw data")
    df.to_csv(config["featurize"]["features_path"], index=False)



if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)