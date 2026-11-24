import pandas as pd

# Load the dataset
df = pd.read_csv('data/survey_regression_data.csv')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Target and updated list of features including match factors and constraints
target_col = 'overall_rating'
feature_cols = [
    'genre_match',
    'vibe_match',
    'size_match',
    'over_budget',
    'crowd_age_gap',
    'different_borough',
    'wait_over_limit_min'
]

print(f"Using Features: {feature_cols}\nTarget: '{target_col}'")

# Drop rows with missing values in these columns for simplicity
data_clean = df[feature_cols + [target_col]].dropna()

X = data_clean[feature_cols]
y = data_clean[target_col]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate and print results
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# Define and train a simple baseline model (always predicts the mean of y_train)
baseline_model = DummyRegressor(strategy='mean')
baseline_model.fit(X_train, y_train)

# Make predictions
y_pred_baseline = baseline_model.predict(X_test)
y_pred_lr = model.predict(X_test)

# Calculate metrics for both
baseline_r2 = r2_score(y_test, y_pred_baseline)
baseline_mse = mean_squared_error(y_test, y_pred_baseline)

lr_r2 = r2_score(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)

print("=== Comparison with Baseline Model ===")
print(f"Baseline (Predict Mean) R^2 Score on Test: {baseline_r2:.4f}")
print(f"Baseline (Predict Mean) MSE on Test:     {baseline_mse:.4f}")
print(f"Linear Regression R^2 Score on Test:       {lr_r2:.4f}")
print(f"Linear Regression MSE on Test:             {lr_mse:.4f}")
