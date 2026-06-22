"""
Training script for California Housing Price Prediction Model

This script:
1. Loads the California Housing dataset
2. Preprocesses the data
3. Trains a RandomForestRegressor model
4. Evaluates model performance
5. Saves the trained model to disk

Usage:
    python src/train.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    N_ESTIMATORS,
    N_JOBS,
    FEATURE_NAMES,
)


def load_data() -> tuple:
    """Load and prepare the California Housing dataset.
    
    Returns:
        Tuple of (X, y) where X contains features and y contains target values
    """
    print("📥 Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {list(df.columns)}")
    
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target values
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"\n📊 Splitting data (test size: {TEST_SIZE*100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train RandomForestRegressor model.
    
    Args:
        X_train: Training features
        y_train: Training target values
        
    Returns:
        Trained model
    """
    print(f"\n🤖 Training RandomForestRegressor...")
    print(f"   n_estimators: {N_ESTIMATORS}")
    print(f"   random_state: {RANDOM_STATE}")
    
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Model training complete")
    
    return model


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n📈 Evaluating model performance...")
    
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
    
    print(f"   Mean Absolute Error (MAE):  ${mae*100000:,.2f}")
    print(f"   Root Mean Squared Error (RMSE): ${rmse*100000:,.2f}")
    print(f"   R² Score: {r2:.4f}")
    
    return metrics


def save_model(model: RandomForestRegressor, path: Path) -> None:
    """Save trained model to disk.
    
    Args:
        model: Trained model
        path: File path to save model
    """
    print(f"\n💾 Saving model to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=0, protocol=4)
    print(f"   ✓ Model saved successfully ({path.stat().st_size / (1024**2):.2f} MB)")


def print_feature_importance(model: RandomForestRegressor, feature_names: list) -> None:
    """Print feature importance ranking.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    """
    print("\n🎯 Feature Importance Ranking:")
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    for idx, row in importance_df.iterrows():
        bar_length = int(row["Importance"] * 50)
        bar = "█" * bar_length
        print(f"   {row['Feature']:<12} {row['Importance']:>6.2%} {bar}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("California Housing Price Prediction - Model Training")
    print("=" * 60)
    
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print feature importance
    print_feature_importance(model, FEATURE_NAMES)
    
    # Save model
    save_model(model, MODEL_PATH)
    
    print("\n" + "=" * 60)
    print("✓ Training pipeline completed successfully!")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
