"""
Data loading and preprocessing for the Customer Transaction Prediction project.

Pipeline (fit only on the training split to avoid leakage):
    raw 200 features
        -> RobustScaler        (200 -> 200, outlier-resilient scaling)
        -> PCA(175)             (200 -> 175, keeps ~most of the variance)
        -> SMOTE                (training fold only, balances ~90/10 target)
"""

import logging

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded dataframe with shape %s", df.shape)
    return df


def get_feature_columns(df: pd.DataFrame):
    """The 200 var_* feature columns, in a stable order."""
    return [f"var_{i}" for i in range(config.N_RAW_FEATURES)]


def split_data(df: pd.DataFrame):
    """Stratified train/test split on the RAW (unscaled) features."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_transform_pipeline(X_train, X_test):
    """
    Fit RobustScaler + PCA on the TRAINING split only, then transform both
    splits. Returns the transformed arrays plus the fitted scaler/PCA so
    they can be persisted for inference.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=config.N_PCA_COMPONENTS, random_state=config.RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    logger.info(
        "PCA: %d components explain %.2f%% of variance",
        config.N_PCA_COMPONENTS,
        pca.explained_variance_ratio_.sum() * 100,
    )
    return X_train_pca, X_test_pca, scaler, pca


def balance_training_data(X_train_pca, y_train):
    """
    Apply SMOTE to the TRAINING fold only (fixes the leakage issue present
    in the original notebook, where SMOTE was applied before the split).
    """
    logger.info("Class distribution before SMOTE: %s", y_train.value_counts().to_dict())
    sm = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train_pca, y_train)
    logger.info("Class distribution after SMOTE: %s", pd.Series(y_res).value_counts().to_dict())
    return X_res, y_res


def run_pipeline(path=config.DATA_PATH):
    df = load_data(path)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_pca, X_test_pca, scaler, pca = fit_transform_pipeline(X_train, X_test)
    X_train_bal, y_train_bal = balance_training_data(X_train_pca, y_train)

    return {
        "X_train": X_train_bal,
        "y_train": y_train_bal,
        "X_test": X_test_pca,
        "y_test": y_test,
        "scaler": scaler,
        "pca": pca,
    }


if __name__ == "__main__":
    data = run_pipeline()
    logger.info("Train shape: %s | Test shape: %s", data["X_train"].shape, data["X_test"].shape)
