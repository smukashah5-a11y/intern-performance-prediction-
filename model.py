import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# ==========================================
# 1. DATA GENERATION / LOADING
# ==========================================
# Creates dataset based on specified requirements:
# - Task completion time
# - Feedback ratings
# - Attendance records
np.random.seed(42)
n_samples = 500

task_completion_time = np.random.uniform(1.0, 10.0, n_samples)  # Hours per task
feedback_ratings = np.random.uniform(1.0, 5.0, n_samples)      # Rating 1 to 5
attendance_records = np.random.uniform(60.0, 100.0, n_samples)  # Attendance %

# Target Variable: Performance Score (Scale 0-100)
performance_score = (
    (10 - task_completion_time) * 3 +
    feedback_ratings * 10 +
    attendance_records * 0.4 +
    np.random.normal(0, 3, n_samples)
)

data = pd.DataFrame({
    'task_completion_time': task_completion_time,
    'feedback_rating': feedback_ratings,
    'attendance_record': attendance_records,
    'performance_score': performance_score
})

# Save synthetic dataset to CSV for submission package
data.to_csv('intern_data.csv', index=False)

# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================
X = data[['task_completion_time', 'feedback_rating', 'attendance_record']]
y = data['performance_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 3. MODEL TRAINING & EVALUATION
# ==========================================
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
}

best_model = None
best_r2 = -float('inf')
best_model_name = ""

print("--- Model Evaluation ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"\nModel: {name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# Save trained best model
joblib.dump(best_model, 'best_intern_model.pkl')
print(f"\nBest Model ('{best_model_name}') saved to best_intern_model.pkl")

# ==========================================
# 4. OUTCOME PREDICTION & CLASSIFICATION
# ==========================================
def predict_intern_outcome(completion_time, rating, attendance):
    model = joblib.load('best_intern_model.pkl')
    sample = np.array([[completion_time, rating, attendance]])
    predicted_score = model.predict(sample)[0]
    
    if predicted_score >= 70:
        outcome = "Likely to Excel"
    elif predicted_score <= 50:
        outcome = "Likely to Struggle"
    else:
        outcome = "Average Performer"
        
    return predicted_score, outcome

# Test sample predictions
score, status = predict_intern_outcome(completion_time=2.0, rating=4.8, attendance=95.0)
print("\n--- Test Sample Outcome ---")
print(f"Predicted Performance Score: {score:.2f}")
print(f"Outcome Classification: {status}")
