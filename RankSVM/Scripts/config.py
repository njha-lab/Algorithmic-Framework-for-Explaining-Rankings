"""
Configuration file for RankSVM project.
"""

from pathlib import Path
import platform

# -------------------------------------------------------
# Project folders
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "Datasets"

SCRIPT_DIR = PROJECT_ROOT / "Scripts"

SVMRANK_DIR = PROJECT_ROOT / "svm_rank"

MODEL_DIR = PROJECT_ROOT / "Models"

PREDICTION_DIR = PROJECT_ROOT / "Predictions"

RESULT_DIR = PROJECT_ROOT / "Results"


# -------------------------------------------------------
# Create output directories if they do not exist
# -------------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Files
# -------------------------------------------------------

TRAIN_FILE = SVMRANK_DIR / "train.dat"

TEST_FILE = SVMRANK_DIR / "test.dat"

MODEL_FILE = MODEL_DIR / "model.dat"

PREDICTION_FILE = PREDICTION_DIR / "predictions.txt"

# -------------------------------------------------------
# RankSVM executables
# -------------------------------------------------------

EXT = ".exe" if platform.system() == "Windows" else ""

LEARN_EXE = SVMRANK_DIR / f"svm_rank_learn{EXT}"
CLASSIFY_EXE = SVMRANK_DIR / f"svm_rank_classify{EXT}"