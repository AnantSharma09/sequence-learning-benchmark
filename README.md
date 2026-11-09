
# Sequence Learning Benchmark: Vanilla RNN vs LSTM

A comparative deep learning benchmark evaluating Vanilla Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks on sequence modeling tasks involving time-series forecasting and text classification.

## Overview

This project investigates how recurrent neural network architectures perform across different sequence learning problems.

The models are evaluated on:

- **Time-Series Forecasting:** Household electricity consumption prediction.
- **Sentiment Classification:** Binary sentiment classification on the IMDb movie review dataset.

The benchmark also includes simpler baseline models to understand whether recurrent architectures provide meaningful improvements.

## Tasks & Models

### 1. Household Electricity Consumption Forecasting

A regression task using historical electricity consumption data to predict future consumption.

Models:
- Vanilla RNN
- LSTM
- Persistence baseline

Evaluation metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### 2. IMDb Sentiment Classification

A binary classification task that predicts whether a movie review expresses positive or negative sentiment.

Models:
- Vanilla RNN
- LSTM
- TF-IDF + Logistic Regression baseline

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Results

### Sentiment Classification

| Model | Accuracy | F1-score |
|---|---:|---:|
| Vanilla RNN | 63.85% | 65.75% |
| LSTM | 75.25% | 73.00% |
| TF-IDF + Logistic Regression | 85.85% | 85.61% |

The classical TF-IDF + Logistic Regression baseline outperformed both recurrent models in this experiment.

### Electricity Forecasting

| Model | RMSE | MAE |
|---|---:|---:|
| Vanilla RNN | 0.5133 | 0.3344 |
| LSTM | 0.5060 | 0.3353 |
| Persistence Baseline | 0.5610 | 0.3472 |

The LSTM achieved the lowest RMSE, while the Vanilla RNN achieved a slightly lower MAE. Both recurrent models outperformed the persistence baseline on these metrics.

*Results are from the current notebook experiment and are not averaged across multiple random seeds.*

## Implementation

The project uses:

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Hugging Face Datasets

The notebook covers data preprocessing, sequence preparation, model construction, training, evaluation, and baseline comparison.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/sequence-learning-benchmark.git
   cd sequence-learning-benchmark
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib scikit-learn torch datasets jupyter
   ```

4. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open the notebook and run the cells sequentially.

## Limitations

- Results are based on a limited experimental setup.
- Models have not been evaluated across multiple random seeds.
- Performance depends on preprocessing, sequence length, and training configuration.
- The benchmark is intended as a learning and comparative experimentation project, not a production forecasting or sentiment analysis system.

## Future Improvements

- Evaluate models across multiple random seeds.
- Experiment with different hidden dimensions and sequence lengths.
- Perform deeper error analysis.
- Explore attention-based sequence models.

