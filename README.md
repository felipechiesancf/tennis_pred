# Octenpus — ATP Match Predictor

End-to-end distributed ML pipeline predicting ATP tennis match outcomes.
Built for IDC5131 (Distributed Computing, Spring 2026) at New College of Florida.

## Live App
https://felipechiesancf.github.io/tennis_pred/

## API
https://web-production-3021d.up.railway.app

## How to Run the Test Script
pip install requests
python test_project.py

## Architecture
- **Ingestion**: Jeff Sackmann GitHub mirror + SofaScore scrape (2016–2025)
- **Processing**: PySpark on Databricks Serverless — Bronze → Silver → Gold medallion layers
- **Features**: 17 window functions (rolling win rates, Elo, H2H, fatigue, serve stats)
- **Model**: Two-stage — XGBoost serve regressors + Markov chain scoring tree. Calibrated with Platt scaling.
- **Results**: 74.4% accuracy / 0.816 AUC on held-out 2025 test set
- **Serving**: FastAPI on Railway + GitHub Pages frontend

## Repo Structure
- notebooks/ — Databricks pipeline notebooks (cleaning, feature engineering, model training)
- models/ — trained XGBoost models (.ubj) + player snapshots
- main.py — FastAPI serving app
- predictor.py — ATPPredictor class (Markov ensemble)
- markov.py — Markov chain scoring model
- index.html — frontend
- test_project.py — end-to-end test script
