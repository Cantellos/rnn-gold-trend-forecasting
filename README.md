# Gold Price Trend Forecasting with RNN

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Recurrent Neural Network implementation for forecasting gold price trends using LSTM architecture and time series analysis.

## Overview

This project implements a deep learning model to predict gold price trends based on historical market data. Using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN), the model learns temporal patterns and dependencies in gold price movements to generate future price forecasts.

**Key Features:**
- LSTM-based RNN architecture for time series forecasting
- Historical gold price data preprocessing and normalization
- Sliding window approach for sequence generation
- Model evaluation with multiple metrics (RMSE, MAE, MAPE)
- Visualization of predicted vs actual prices
- Backtesting capabilities

## Motivation

Gold prices are influenced by numerous factors including economic indicators, geopolitical events, currency fluctuations, and market sentiment. Accurate price forecasting can provide valuable insights for:
- **Investors**: Informed decision-making for portfolio management
- **Traders**: Identification of entry and exit points
- **Financial Analysts**: Understanding market trends and patterns
- **Risk Management**: Hedging strategies and exposure control

## Model Architecture

### LSTM Network Structure

```
Input Layer (Sequence Length × Features)
    ↓
LSTM Layer 1 (units=50, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (units=50, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 3 (units=50)
    ↓
Dropout (0.2)
    ↓
Dense Layer (units=1)
    ↓
Output (Predicted Price)
```

### Hyperparameters

- **Sequence Length**: 60 days (lookback window)
- **Batch Size**: 32
- **Epochs**: 100
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Dropout Rate**: 0.2 (prevents overfitting)

## Dataset

The model is trained on historical gold price data including:
- **Daily closing prices**
- **Trading volume**
- **High/Low prices**
- **Open prices**

**Data Source**: [Specify your data source - Yahoo Finance, Quandl, etc.]

**Time Period**: [Specify date range - e.g., 1990-2024]

## Project Structure

```
rnn-gold-trend-forecasting/
├── data/
│   ├── raw/                 # Raw gold price data
│   └── processed/           # Preprocessed data
├── models/
│   ├── lstm_model.py        # LSTM model architecture
│   └── saved_models/        # Trained model checkpoints
├── notebooks/
│   ├── EDA.ipynb            # Exploratory Data Analysis
│   ├── training.ipynb       # Model training
│   └── evaluation.ipynb     # Results visualization
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── prediction.py
│   └── evaluation.py
├── results/
│   ├── plots/               # Visualization outputs
│   └── metrics/             # Performance metrics
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Cantellos/rnn-gold-trend-forecasting.git
cd rnn-gold-trend-forecasting
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries

```txt
tensorflow>=2.12.0
keras>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
yfinance>=0.2.0  # If using Yahoo Finance
```

## Usage

### 1. Data Preparation

```python
from src.data_preprocessing import load_data, preprocess_data

# Load raw gold price data
df = load_data('data/raw/gold_prices.csv')

# Preprocess and create sequences
X_train, y_train, X_test, y_test, scaler = preprocess_data(
    df, 
    sequence_length=60,
    train_split=0.8
)
```

### 2. Model Training

```python
from src.model_training import build_lstm_model, train_model

# Build LSTM model
model = build_lstm_model(
    sequence_length=60,
    n_features=1,
    lstm_units=[50, 50, 50],
    dropout_rate=0.2
)

# Train the model
history = train_model(
    model,
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1
)
```

### 3. Generate Predictions

```python
from src.prediction import predict_prices

# Make predictions
predictions = predict_prices(model, X_test, scaler)

# Visualize results
from src.evaluation import plot_predictions
plot_predictions(y_test, predictions, save_path='results/plots/forecast.png')
```

### 4. Model Evaluation

```python
from src.evaluation import calculate_metrics

# Calculate performance metrics
metrics = calculate_metrics(y_test, predictions)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"MAPE: {metrics['mape']:.2f}%")
```

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| RMSE   | [Your value] |
| MAE    | [Your value] |
| MAPE   | [Your value] |
| R²     | [Your value] |


## Model Improvements

Potential enhancements:
- [ ] Multivariate analysis (incorporate economic indicators, USD index, inflation rates)
- [ ] Attention mechanisms for better long-term dependencies
- [ ] Bidirectional LSTM for forward and backward temporal context
- [ ] Ensemble methods combining multiple models
- [ ] Real-time data streaming and continuous learning
- [ ] Integration with sentiment analysis from news sources
- [ ] Hyperparameter tuning with GridSearch or Bayesian optimization

## Limitations

- **Market volatility**: Extreme events (e.g., geopolitical crises) are difficult to predict
- **Data dependency**: Model performance relies on historical patterns continuing
- **Lagging indicators**: Technical analysis may not capture fundamental shifts
- **Overfitting risk**: Complex models may learn noise rather than signal

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Commit changes: `git commit -m "Add improvement"`
4. Push to branch: `git push origin feature/improvement`
5. Submit a pull request

## Dataset
- [List your data sources]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow/Keras team for deep learning framework
- Financial data providers
- Open-source community

## Contact

For questions or collaboration opportunities, please open an issue on the [GitHub repository](https://github.com/Cantellos/rnn-gold-trend-forecasting).

---

**Disclaimer**: This model is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.
