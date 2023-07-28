import sys
from pathlib import Path
import pandas as pd
import dvc.api

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


def data_load() -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    config = dvc.api.params_show()

    logger = get_logger("DATA_LOAD", log_level=config["base"]["log_level"])

    logger.info("Get dataset")
    df = pd.read_csv(config["data"]["data_load"])
    logger.info("Clean dataset")
    logger.info("Drop missing values")
    df = df.dropna()
    logger.info("Drop CUST_ID column")
    df = df.drop('CUST_ID', axis=1)
    logger.info("Remove outliers")
    df = remove_outliers(df)
    logger.info("Save raw data")
    df.to_csv(config["data"]["data_preprocessing"], index=False)



if __name__ == "__main__":
    data_load()