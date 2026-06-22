"""
Train and save the California Housing Price prediction model.
Run this script to generate housing_model.pkl before starting the app.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model():
    """Train RandomForestRegressor on California Housing dataset."""
    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    print("Preparing data for training...")
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate model
    r2_score = model.score(X_test, y_test)
    print(f"Model R² Score: {r2_score:.4f}")

    print("Saving model to housing_model.pkl...")
    joblib.dump(model, "housing_model.pkl", compress=0, protocol=4)
    print("✓ Model training complete!")

if __name__ == "__main__":
    train_model()
