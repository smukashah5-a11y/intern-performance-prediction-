# Intern Performance Prediction Model

## Project Overview
This machine learning regression model predicts intern performance and categorizes whether an intern is likely to excel or struggle based on historical performance indicators.

## Features Used
- **Task Completion Time**: Average hours spent on assigned tasks.
- **Feedback Ratings**: Average supervisor evaluation score (1–5 scale).
- **Attendance Records**: Total percentage of attendance.

## Model Details
- **Algorithms Evaluated**: Random Forest Regressor & XGBoost Regressor
- **Target Variable**: Continuous Performance Score (0–100 scale)

## Outcome Classification Rules
- **Score >= 70**: Likely to Excel
- **Score 51 – 69**: Average Performer
- **Score <= 50**: Likely to Struggle

## Requirements & Usage
To run the project locally, install dependencies and execute the script:

```bash
pip install pandas numpy scikit-learn xgboost joblib
python model.py

