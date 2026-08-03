"""
Central configuration for the Customer Transaction Prediction project.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Kaggle "Santander Customer Transaction Prediction" dataset.
# Not included in this repo due to size — download it and place it here.
DATA_PATH = DATA_DIR / "Customer-Transactions.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
ID_COL = "ID_code"
N_RAW_FEATURES = 200          # var_0 .. var_199
N_PCA_COMPONENTS = 175        # chosen from the explained-variance curve

RANDOM_STATE = 42
TEST_SIZE = 0.30
