# XAU/USD Gold Price Forecasting with Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Comparative study of neural network architectures for XAU/USD (Gold/USD) price trend forecasting using Multi-Layer Perceptrons (MLP) and Recurrent Neural Networks (RNN).

**Project Work - Deep Learning**  
Andrea Cantelli - Matricola: 156529

## Overview

This project implements and compares multiple neural network architectures to predict gold price movements, analyzing both single-step and multi-step forecasting approaches. The study demonstrates that while MLPs perform better for single-step predictions, RNNs provide valuable multi-step forecasting capabilities essential for market analysis and trading applications.

## Dataset

**Source:** [XAU/USD Gold Price Historical Data (2004-2025)](https://www.kaggle.com) - Kaggle

**Features:**
- `Date`: Timestamp
- `Open`: Opening price
- `High`: Daily high price
- `Low`: Daily low price
- `Close`: Closing price
- `Volume`: Trading volume

**Period:** 2004-2025 (21 years of historical data)

## Data Preparation

### Preprocessing Steps

1. **Data Cleaning**
   - Removal of rows with null values
   - Removal of `Date` column (converted to sequential index)

2. **Feature Engineering**
   
   Added technical financial indicators:
   - **Moving Average (MA)**: Trend identification
   - **Exponential Moving Average (EMA)**: Weighted recent prices
   - **Stochastic Oscillator**: Momentum indicator
   - **Relative Strength Index (RSI)**: Overbought/oversold conditions

3. **Data Splitting**
   
   Holdout cross-validation strategy:
   - **70% Training set**: Model learning
   - **15% Validation set**: Hyperparameter tuning
   - **15% Test set**: Final evaluation

4. **Target Variable**
   - `future_close`: Future closing price for prediction

### Normalization

MinMax scaler performed better as the gold price dataset is uniformly distributed without significant outliers.

### Sliding Window for RNN

Sequential sliding-windows implementation:
- Extracts overlapping input-target pairs from time series
- Preserves chronological order (no shuffling)
- Maintains temporal correlations

**Configuration:**

| Model | Input Length | Output Length | Type |
|-------|-------------|---------------|------|
| **RNN_single** | 7 timesteps | 1 timestep | Single-step prediction |
| **RNN_multi** | 30 timesteps | 7 timesteps | Multi-step prediction |

## Model Architectures

### 1. MLP (Multi-Layer Perceptron)

Feedforward neural networks with fully connected layers.

#### MLP1: Single Hidden Layer

```
Input Layer (All features from single timeframe)
    ↓
Hidden Layer (64 units, ELU activation)
    ↓
Output Layer (1 unit - next closing price)
```

**Configuration:**
- Hidden size: 64
- Activation: ELU (Exponential Linear Unit)
- Batch size: 32

**Performance:**
- Test MSE: 16.35
- Accuracy (1% tolerance): **~68%**
- Average error: 0.813%

#### MLP2: Two Hidden Layers

```
Input Layer
    ↓
Hidden Layer 1 (64 units, ReLU activation)
    ↓
Hidden Layer 2 (32 units, ReLU activation)
    ↓
Output Layer (1 unit)
```

**Configuration:**
- Hidden sizes: 64, 32
- Activation: ReLU
- Batch size: 32

**Performance:**
- Test MSE: 446.35
- Accuracy (1% tolerance): **~70%** ✅ Best overall
- Average error: 0.778%

### 2. RNN (Recurrent Neural Network)

Networks designed for sequential data with temporal dependencies.

#### RNN_single: Single-Step Prediction

Predicts the next single closing price based on 7 previous timesteps.

**Configuration:**
- Hidden size: 64
- Layers: 1
- Batch size: 128
- Learning rate: 0.0007

**Performance:**
- Test MSE: 0.000325
- Accuracy (1% tolerance): **~42%**
- Average error: 1.423%

#### RNN_multi: Multi-Step Prediction

Predicts 7 future closing prices based on 30 previous timesteps.

**Configuration:**
- Hidden size: 64
- Layers: 1
- Batch size: 128
- Learning rate: 0.0006

**Performance:**
- Test MSE: 0.00145
- Individual predictions accuracy: ~24%
- **Averaged predictions accuracy: ~41%** (smoothed predictions)
- Average error: 1.676% (individual), 1.676% (averaged)

## Training Configuration

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | 500 | With early stopping |
| **Patience** | 30 | For early stopping |
| **Learning Rate** | 0.001 (MLP), 0.0006-0.0007 (RNN) | Optimized per model |
| **Batch Size** | 32 (MLP), 128 (RNN) | |

### Loss Functions Tested

- **MSE** (Mean Squared Error): Standard regression loss
- **SmoothL1** (Huber Loss): Robust to outliers (PyTorch)
- **MAPE** (Mean Absolute Percentage Error): Percentage-based metric

### Optimizers

- **Adam** ✅: Best performance across all models
- **RMSprop**: Poor performance with RNNs

### Regularization

- **Early Stopping**: Prevents overfitting (patience 30)
- **Model Checkpointing**: Saves best model weights based on validation loss

## Evaluation Metrics

Three complementary metrics for comprehensive evaluation:

1. **MSE/Huber Loss on Test Set**: Quantifies prediction error magnitude
2. **Accuracy-based Metric**: Percentage of predictions within 1% tolerance of actual values
3. **Average Percentage Error**: Mean error as percentage of actual price

## Results Summary

| Model | Architecture | Accuracy (1%) | MSE Test | Avg Error |
|-------|-------------|---------------|----------|-----------|
| **MLP1** | 1 layer (64) | 68% | 16.35 | 0.813% |
| **MLP2** ✅ | 2 layers (64, 32) | **70%** | 446.35 | **0.778%** |
| **RNN_single** | 1 layer (64) | 42% | 0.000325 | 1.423% |
| **RNN_multi** | 1 layer (64) | 41% (avg) | 0.00145 | 1.676% |

### Key Findings

1. **MLPs outperform RNNs for single-step predictions** (~70% vs ~42% accuracy)
   - Lower computational cost
   - Better accuracy
   - More suitable for next-day price forecasting

2. **RNNs struggle with sudden trend changes** and volatile market movements
   - Performance degrades with prediction horizon
   - Better for trend following than volatility prediction

3. **Multi-step RNN predictions are valuable despite lower accuracy**
   - Provide week-ahead forecasts (7 days)
   - More practical for trading strategy development
   - Averaged predictions smooth out noise

4. **Adam optimizer consistently superior** to RMSprop across all architectures

5. **Increasing training epochs beyond optimal point** risks overfitting without performance gains

## Project Structure

```
rnn-gold-trend-forecasting/
├── data/
│   ├── raw/                          # Original XAU/USD dataset
│   └── processed/                    # Preprocessed data with technical indicators
├── models/
│   ├── mlp_models.py                 # MLP1 and MLP2 implementations
│   ├── rnn_models.py                 # RNN_single and RNN_multi
│   └── saved_models/                 # Trained model checkpoints
├── notebooks/
│   ├── 01_data_preparation.ipynb     # Data cleaning and feature engineering
│   ├── 02_mlp_training.ipynb         # MLP experiments
│   ├── 03_rnn_training.ipynb         # RNN experiments
│   └── 04_evaluation.ipynb           # Results comparison
├── src/
│   ├── data_preprocessing.py         # Data loading and normalization
│   ├── feature_engineering.py        # Technical indicators calculation
│   ├── sliding_window.py             # Time series window generation
│   ├── training.py                   # Training loops with early stopping
│   └── evaluation.py                 # Metrics calculation
├── results/
│   ├── plots/                        # Training curves and predictions
│   └── metrics/                      # Performance comparisons
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Cantellos/rnn-gold-trend-forecasting.git
cd rnn-gold-trend-forecasting
```

2. **Create virtual environment**
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
torch>=2.0.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
ta>=0.10.0  # Technical analysis indicators
```

## Usage

### 1. Data Preparation

```python
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import add_technical_indicators

# Load dataset
df = load_data('data/raw/xauusd_2004_2025.csv')

# Add technical indicators
df = add_technical_indicators(df)  # MA, EMA, RSI, Stochastic

# Preprocess and split
train_data, val_data, test_data = preprocess_data(
    df, 
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    normalize=True,
    scaler='minmax'
)
```

### 2. Train MLP Model

```python
from models.mlp_models import MLP2
from src.training import train_model

# Initialize model
model = MLP2(
    input_size=10,  # Number of features
    hidden_size1=64,
    hidden_size2=32,
    output_size=1
)

# Train with early stopping
history = train_model(
    model,
    train_data,
    val_data,
    epochs=500,
    batch_size=32,
    learning_rate=0.001,
    optimizer='adam',
    loss_fn='mse',
    patience=30
)
```

### 3. Train RNN Model

```python
from models.rnn_models import RNN_multi
from src.sliding_window import create_sequences

# Create sliding window sequences
X_train, y_train = create_sequences(
    train_data,
    input_length=30,
    output_length=7
)

# Initialize RNN
model = RNN_multi(
    input_size=10,
    hidden_size=64,
    output_size=7,
    num_layers=1
)

# Train
history = train_model(
    model,
    (X_train, y_train),
    (X_val, y_val),
    epochs=500,
    batch_size=128,
    learning_rate=0.0006,
    optimizer='adam',
    loss_fn='smoothl1',
    patience=30
)
```

### 4. Evaluate and Predict

```python
from src.evaluation import calculate_metrics, plot_predictions

# Make predictions
predictions = model.predict(X_test)

# Calculate metrics
metrics = calculate_metrics(y_test, predictions, tolerance=0.01)
print(f"Accuracy (1% tolerance): {metrics['accuracy']:.2%}")
print(f"MSE: {metrics['mse']:.6f}")
print(f"Average % Error: {metrics['avg_pct_error']:.3%}")

# Visualize results
plot_predictions(y_test, predictions, save_path='results/plots/predictions.png')
```

## Key Insights

### When to Use Each Model

**Use MLP when:**
- Need next-day price prediction (single-step)
- Computational efficiency is important
- Higher accuracy is priority
- Features are comprehensive (technical indicators)

**Use RNN when:**
- Need multi-day forecasts (week ahead)
- Sequential dependencies are important
- Trading strategy requires trend forecasting
- Averaged predictions acceptable

### Limitations

1. **RNN Performance**: Lower accuracy compared to MLP for single-step predictions
2. **Volatile Markets**: All models struggle with sudden trend reversals
3. **Data Dependency**: Performance relies on historical patterns continuing
4. **Training Time**: RNNs require significantly longer training

### Best Practices

1. **Feature Engineering is Critical**: Technical indicators significantly improve performance
2. **MinMax Normalization**: Works better than Standard Scaler for financial data
3. **No Shuffling for Time Series**: Preserving temporal order is essential
4. **Early Stopping**: Prevents overfitting, optimal patience = 30 epochs
5. **Adam Optimizer**: Consistently outperforms RMSprop

## Future Developments

### Planned Enhancements

1. **Enhanced Feature Engineering**
   - US Federal Reserve interest rates
   - US inflation rates (CPI)
   - US unemployment data
   - USD index
   - Global economic indicators

2. **Extended Asset Coverage**
   - S&P 500 index
   - NASDAQ composite
   - Silver (XAG/USD)
   - Other precious metals
   - Cross-asset correlation analysis

3. **Advanced Architectures**
   - **LSTM** (Long Short-Term Memory): Better long-term dependencies
   - **GRU** (Gated Recurrent Unit): More efficient than LSTM
   - **Attention Mechanisms**: Focus on relevant time periods
   - **Transformer Models**: State-of-the-art sequence modeling

4. **Ensemble Methods**
   - Combine MLP and RNN predictions
   - Weighted averaging based on market conditions
   - Model confidence estimation

## Academic Context

This project was developed as a Deep Learning course project work, demonstrating:
- Implementation of multiple neural network architectures
- Comparative analysis of forecasting approaches
- Real-world application to financial time series
- Rigorous experimental methodology with proper train/val/test splits

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: Kaggle
- Framework: PyTorch
- Technical indicators: TA-Lib / ta library
- Inspiration: Financial time series forecasting literature

## Contact

For questions or collaboration opportunities, please open an issue on the [GitHub repository](https://github.com/Cantellos/rnn-gold-trend-forecasting).

---

## Disclaimer

This model is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Financial markets are inherently unpredictable, and past performance does not guarantee future results. Always consult with financial professionals before making investment choices.
