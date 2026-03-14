# AI Crypto Prediction System

AI-powered cryptocurrency forecasting platform that generates rolling 14-day price predictions using ensemble machine learning models and provides results through a FastAPI backend and an interactive dashboard.

---

## Overview

This project is an end-to-end cryptocurrency forecasting system designed to analyze historical market data, generate predictive features, train machine learning models, and serve predictions through an API and visualization dashboard.

The system uses multiple machine learning models and combines their outputs to generate more stable forecasts.

The architecture is designed to be modular, allowing easy integration of additional models, features, and data sources.

---

## Features

- 14-day rolling price forecasts
- Ensemble machine learning models
- REST API for predictions
- Interactive dashboard for visualization
- Feature engineering pipeline
- Model evaluation and comparison
- Modular architecture for easy extension

---

## Machine Learning Models

The system currently includes the following models:

- Random Forest
- XGBoost
- LightGBM

These models are combined in an ensemble approach to produce more robust predictions.

---

## System Architecture

The project follows a modular architecture where each stage of the machine learning pipeline is separated into dedicated components.

Main components:


data → feature engineering → model training → prediction → API → dashboard


Detailed architecture documentation is available in:


ARCHITECTURE.md


---

## Project Structure


ai-crypto-prediction-system/

api/ FastAPI application and API endpoints
config/ Configuration and settings
dashboard/ Frontend dashboard for visualization
data/ Data collection utilities
evaluation/ Model evaluation logic
features/ Feature engineering pipeline
models/ Machine learning models
prediction/ Prediction and forecasting logic
scripts/ Training and pipeline scripts
tests/ Unit tests
utils/ Utility functions

requirements.txt Python dependencies
ARCHITECTURE.md Architecture documentation
.env.example Environment variable template


---

## Installation

Clone the repository:

```bash
git clone https://github.com/MuhammedCanOzkesemen/ai-crypto-prediction-system.git
cd ai-crypto-prediction-system

Install dependencies:

pip install -r requirements.txt

Create an environment file:

.env

Example:

COINGECKO_API_KEY=your_api_key_here
Running the API

Start the FastAPI server:

python -m uvicorn api.app:app --host 127.0.0.1 --port 8010

API documentation will be available at:

http://127.0.0.1:8010/docs
Dashboard

The project includes an interactive dashboard for visualizing predictions and historical data.

Run the API and open:

http://127.0.0.1:8010/dashboard

The dashboard displays:

Historical market data

Forecast intervals

Model agreement

14-day forecast path

Security

Sensitive data such as API keys are stored in environment variables.

The .env file is excluded from version control using .gitignore.

Example template is provided in:

.env.example
Future Improvements

Possible future extensions:

Additional machine learning models

Deep learning models (LSTM / Transformers)

Automated data refresh pipeline

Model retraining automation

Deployment to cloud infrastructure

Real-time prediction updates

License

This project is licensed under the MIT License.
