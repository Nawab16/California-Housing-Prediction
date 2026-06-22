# California Housing Price Prediction

A professional machine learning application that predicts California housing prices using RandomForest regression with a Streamlit web interface.

## 🎯 Project Overview

This project implements an end-to-end machine learning solution for predicting median house values in California based on demographic and geographic features from the 1990 census dataset.

**Model Performance**: R² Score = 0.81  
**Dataset**: California Housing Dataset (~20,640 samples, 8 features)

## 📁 Project Structure

```
California-Housing-Prediction/
├── src/                          # Source code directory
│   ├── main.py                   # Streamlit web application
│   ├── train.py                  # Model training script
│   └── config.py                 # Configuration settings
├── notebooks/                    # Jupyter notebooks
│   └── california_housing_analysis.ipynb
├── data/                         # Data directory (for datasets)
├── models/                       # Trained models (auto-generated)
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository** (if applicable):
```bash
git clone https://github.com/Nawab16/California-Housing-Prediction.git
cd California-Housing-Prediction
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Training the Model

Before running the application, train the model:

```bash
python src/train.py
```

This script will:
- Load the California Housing dataset
- Split data into training/testing sets (80/20)
- Train a RandomForestRegressor model
- Evaluate model performance
- Save the model to `models/housing_model.pkl`

### Running the Application

Start the Streamlit web application:

```bash
streamlit run src/main.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Features

### Input Features
1. **Median Income** - Median income in the block group (in $10,000 units)
2. **House Age** - Median age of houses in years
3. **Average Rooms** - Average number of rooms per household
4. **Average Bedrooms** - Average number of bedrooms per household
5. **Population** - Total population in the block group
6. **Average Occupancy** - Average occupancy per household
7. **Latitude** - Geographic latitude coordinate
8. **Longitude** - Geographic longitude coordinate

### Prediction Output
- Predicted median house value in US dollars
- Model confidence metrics (R² Score)

## 🤖 Model Details

### Algorithm
**RandomForestRegressor**
- Number of trees: 100
- Random state: 42 (for reproducibility)
- Multi-threading: Enabled (n_jobs=-1)

### Performance Metrics
```
Mean Absolute Error (MAE):     ~$70,000
Root Mean Squared Error (RMSE): ~$92,000
R² Score:                       0.81
```

## 📈 Feature Importance
The model identifies the most influential features for price prediction:
1. Median Income
2. House Age
3. Latitude
4. Population
5. Average Rooms
6. Average Occupancy
7. Longitude
8. Average Bedrooms

## 🛠️ Technology Stack

- **Python 3.8+** - Programming language
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **joblib** - Model serialization

## 📝 Configuration

Edit `src/config.py` to customize:
- Model hyperparameters (n_estimators, random_state)
- Train/test split ratio
- Model save path
- Feature names and display labels

## 🧪 Example Usage

### Using the Web Interface
1. Open the Streamlit app: `streamlit run src/main.py`
2. Enter housing features in the sidebar
3. Click "🔮 Predict Price"
4. View the predicted price and model information

### Example Prediction Input
| Feature | Value |
|---------|-------|
| Median Income | $8.30 |
| House Age | 41 years |
| Average Rooms | 6.98 |
| Average Bedrooms | 1.02 |
| Population | 322 |
| Average Occupancy | 2.56 |
| Latitude | 37.88 |
| Longitude | -122.23 |

**Predicted Output**: ~$450,000

## 📚 Notebooks

The `notebooks/california_housing_analysis.ipynb` contains:
- Data exploration and visualization
- Feature correlation analysis
- Model training and evaluation
- Performance metrics and analysis

## 🔄 Workflow

```
1. Data Loading (fetch_california_housing)
   ↓
2. Data Preprocessing & Splitting
   ↓
3. Model Training (RandomForest)
   ↓
4. Model Evaluation & Feature Importance
   ↓
5. Model Persistence (joblib.dump)
   ↓
6. Web Deployment (Streamlit App)
```

## ⚙️ Deployment

### Local Deployment
```bash
streamlit run src/main.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from GitHub repository

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/main.py"]
```

## 📋 Requirements

See `requirements.txt` for full dependencies:
- streamlit>=1.0.0
- scikit-learn>=1.0.0
- pandas>=1.3.0
- numpy>=1.21.0
- joblib>=1.1.0

## 🐛 Troubleshooting

### Model not found error
**Solution**: Run `python src/train.py` to train and save the model

### Import errors
**Solution**: Install all requirements: `pip install -r requirements.txt`

### Port already in use
**Solution**: Change Streamlit port: `streamlit run src/main.py --server.port 8502`

## 📖 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

## 👤 Author

**Nawab16** - [GitHub Profile](https://github.com/Nawab16)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⭐ Acknowledgments

- California Housing Dataset from StatLib
- Built with Streamlit and scikit-learn
- Inspired by machine learning best practices

---

**Last Updated**: June 2026  
**Status**: ✅ Production Ready