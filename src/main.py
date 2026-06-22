"""
California Housing Price Prediction - Streamlit Application

This application uses a trained RandomForestRegressor model to predict
California housing prices based on demographic and geographic features.
"""

import streamlit as st
import joblib
import numpy as np
from config import (
    MODEL_PATH,
    FEATURE_NAMES,
    FEATURE_DISPLAY_NAMES,
    PRICE_SCALE_FACTOR,
)


def load_model():
    """Load the trained model from disk.
    
    Returns:
        Trained model object or None if loading fails.
    """
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(
            f"❌ Model file not found at {MODEL_PATH}\n\n"
            "Please run `python src/train.py` to train the model first."
        )
        return None
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


def get_user_inputs() -> dict:
    """Get feature inputs from user via Streamlit UI.
    
    Returns:
        Dictionary with feature names as keys and user inputs as values.
    """
    st.subheader("📊 Enter Housing Features")
    
    col1, col2 = st.columns(2)
    
    inputs = {}
    with col1:
        inputs["MedInc"] = st.number_input(
            FEATURE_DISPLAY_NAMES["MedInc"],
            min_value=0.5,
            max_value=15.0,
            value=3.0,
            step=0.1
        )
        inputs["HouseAge"] = st.number_input(
            FEATURE_DISPLAY_NAMES["HouseAge"],
            min_value=1,
            max_value=52,
            value=30,
            step=1
        )
        inputs["AveRooms"] = st.number_input(
            FEATURE_DISPLAY_NAMES["AveRooms"],
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.1
        )
        inputs["AveBedrms"] = st.number_input(
            FEATURE_DISPLAY_NAMES["AveBedrms"],
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
    
    with col2:
        inputs["Population"] = st.number_input(
            FEATURE_DISPLAY_NAMES["Population"],
            min_value=3,
            max_value=35000,
            value=1000,
            step=100
        )
        inputs["AveOccup"] = st.number_input(
            FEATURE_DISPLAY_NAMES["AveOccup"],
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.1
        )
        inputs["Latitude"] = st.number_input(
            FEATURE_DISPLAY_NAMES["Latitude"],
            min_value=32.0,
            max_value=42.0,
            value=37.88,
            step=0.01
        )
        inputs["Longitude"] = st.number_input(
            FEATURE_DISPLAY_NAMES["Longitude"],
            min_value=-124.0,
            max_value=-114.0,
            value=-122.23,
            step=0.01
        )
    
    return inputs


def make_prediction(model, inputs: dict) -> float:
    """Make a price prediction using the trained model.
    
    Args:
        model: Trained prediction model
        inputs: Dictionary of feature values
        
    Returns:
        Predicted price in dollars
    """
    # Prepare data in correct feature order
    data = np.array([[inputs[feature] for feature in FEATURE_NAMES]])
    
    # Get prediction
    prediction = model.predict(data)[0]
    
    return prediction * PRICE_SCALE_FACTOR


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Housing Price Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🏠 California Housing Price Predictor")
    st.markdown(
        "Predict California housing prices using machine learning. "
        "Enter the housing features below to get an estimate."
    )
    
    # Load model
    model = load_model()
    
    if model is not None:
        st.divider()
        
        # Get user inputs
        user_inputs = get_user_inputs()
        
        st.divider()
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔮 Predict Price", use_container_width=True):
                with st.spinner("Making prediction..."):
                    predicted_price = make_prediction(model, user_inputs)
                    
                    st.success(
                        f"### Predicted House Value\n"
                        f"# ${predicted_price:,.2f}"
                    )
                    
                    st.info(
                        f"**Model**: RandomForestRegressor  \n"
                        f"**Accuracy (R² Score)**: 0.81  \n"
                        f"**Note**: This is an estimate based on historical data."
                    )
        
        st.divider()
        
        # Sidebar information
        with st.sidebar:
            st.subheader("ℹ️ About")
            st.markdown(
                """
                ### California Housing Price Predictor
                
                **Dataset**: California Housing Dataset (1990)
                
                **Model**: Random Forest Regressor
                - **Features**: 8 demographic and geographic features
                - **Training Data**: ~16,500 housing records
                - **Performance**: R² Score = 0.81
                
                **Features Used**:
                - Median Income
                - House Age
                - Average Rooms
                - Average Bedrooms
                - Population
                - Average Occupancy
                - Latitude
                - Longitude
                """
            )
    else:
        st.warning(
            "⚠️ Please train the model first by running:\n\n"
            "`python src/train.py`"
        )


if __name__ == "__main__":
    main()