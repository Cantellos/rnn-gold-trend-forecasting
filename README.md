# Gold Price Trend Forecasting via Deep Learning

This repository provides a robust framework for predicting gold price movements using Recurrent Neural Networks (RNNs). The primary objective is to leverage Long Short-Term Memory (LSTM) architectures to model the inherent volatility and temporal dependencies found in precious metal market data.

## Project Overview
Financial time-series data is characterized by non-stationarity and noise. This project implements a standardized pipeline to transform raw historical price data into a supervised learning format, followed by the training of a predictive model capable of identifying trend directionality.

## Key Features
- **Data Engineering:** Implementation of sliding window techniques to create temporal sequences for training.
- **Neural Architecture:** Optimized LSTM layers designed to prevent vanishing gradient issues in long sequences.
- **Evaluation Metrics:** Performance assessment using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
- **Visualization:** Integrated reporting tools for comparing predicted trends against actual market closing prices.

## Technical Stack
- **Language:** Python 3.x
- **Deep Learning:** TensorFlow / Keras
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib

## Repository Structure
- `data/`: Historical datasets and source files.
- `notebooks/`: Exploratory Data Analysis (EDA) and model prototyping.
- `src/`: Core Python modules for data processing and model definitions.
- `requirements.txt`: Configuration file for environment reproducibility.

## Installation and Setup
1. Clone the repository:
```bash
   git clone [https://github.com/Cantellos/gold-price-forecasting.git](https://github.com/Cantellos/gold-price-forecasting.git)
```
   
## Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Execute the training pipeline:
```bash
python src/train_model.py
```

## Disclaimer
This project is for educational and research purposes only. It does not constitute financial advice. Market investments carry inherent risks, and past performance is not indicative of future results.
