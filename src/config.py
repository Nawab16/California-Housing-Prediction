"""
Configuration settings for California Housing Price Prediction application.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_PATH = MODELS_DIR / "housing_model.pkl"
MODEL_NAME = "RandomForestRegressor"
MODEL_VERSION = "1.0"

# Training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
N_JOBS = -1

# Feature names in order
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

# Feature display names for UI
FEATURE_DISPLAY_NAMES = {
    "MedInc": "Median Income ($10k)",
    "HouseAge": "House Age (years)",
    "AveRooms": "Average Rooms",
    "AveBedrms": "Average Bedrooms",
    "Population": "Population",
    "AveOccup": "Average Occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude"
}

# Prediction scaling
PRICE_SCALE_FACTOR = 100000  # Model outputs in units of $100k
