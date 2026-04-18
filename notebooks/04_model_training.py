
"""
Toronto Island Ferry Ticket Demand - Model Training & Evaluation
Implements Baseline, ML, and Time-Series models for forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
# filepath: 04_model_training.py
try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Run 'pip install xgboost' and restart.")
    exit(1)


# Time series imports
# filepath: 04_model_training.py
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    print("statsmodels not installed. Run 'pip install statsmodels' and restart.")
    exit(1)
# ...existing code...
#from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

import os
os.makedirs('output/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 70)
print("SHORT-TERM DEMAND FORECASTING - MODEL TRAINING & EVALUATION")
print("=" * 70)

# Load the feature-engineered dataset
df = pd.read_csv('data/ferry_tickets_engineered.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

print(f"\n📊 Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")

# ============================================================================
# DEFINE EVALUATION METRICS
# ============================================================================
def calculate_metrics(y_true, y_pred, name="Model"):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Handle zero values for MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = np.nan
    
    # Calculate within-threshold accuracy
    threshold_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
    
    return {
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'Within ±10 (%)': threshold_10
    }

# ============================================================================
# TRAIN-TEST SPLIT (Time-based)
# ============================================================================
print("\n🔧 Performing time-based train-test split...")

# Use last 20% of data for testing (time-based split)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"Training set: {len(train_df)} records ({train_df['Timestamp'].min()} to {train_df['Timestamp'].max()})")
print(f"Test set: {len(test_df)} records ({test_df['Timestamp'].min()} to {test_df['Timestamp'].max()})")

# Define feature columns - only numeric features
# IMPORTANT: Exclude all target variables to prevent data leakage
exclude_cols = ['_id', 'Timestamp', 'Sales Count', 'Redemption Count', 'TimeOfDay', 'Date', 'DayName', 
                'time_diff', 'index']
feature_cols = [c for c in df.columns if c not in exclude_cols]
# Exclude all target columns (these are what we're trying to predict)
feature_cols = [c for c in feature_cols if 'Target' not in c]
# Exclude Redemption-related features (we're predicting Sales)
feature_cols = [c for c in feature_cols if not c.startswith('Redemption')]

# Ensure only numeric columns
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
feature_cols = numeric_features

print(f"\nFeatures for training: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:10]}")

# Verify no target leakage
target_cols_in_features = [c for c in feature_cols if 'Target' in c]
if target_cols_in_features:
    print(f"WARNING: Target columns found in features: {target_cols_in_features}")
else:
    print("✓ No target columns in features (data leakage check passed)")

# ============================================================================
# BASELINE MODELS
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING BASELINE MODELS")
print("=" * 70)

results = []

# Target variable (forecasting Sales 1-hour ahead)
target_col = 'Sales_Target_1hr'

# Prepare data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

# Handle any NaN/Inf in features
from sklearn.impute import SimpleImputer

# Replace inf with nan, then fill
X_train = np.where(np.isinf(X_train), np.nan, X_train)
X_test = np.where(np.isinf(X_test), np.nan, X_test)

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Impute any NaN in target
y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)
y_test = np.nan_to_num(y_test, nan=0, posinf=0, neginf=0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# 1. Naive Forecast (use last value)
print("\n📊 Training Naive Forecast...")
y_pred_naive = test_df['Sales Count'].values  # Predict same as current
results.append(calculate_metrics(y_test, y_pred_naive, "Naive (t=t-1)"))

# 2. Moving Average (last 4 intervals = 1 hour)
print("📊 Training Moving Average...")
y_pred_ma = test_df['Sales Count'].rolling(window=4, min_periods=1).mean().values
results.append(calculate_metrics(y_test, y_pred_ma, "Moving Average (1hr)"))

# 3. Linear Regression
print("📊 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_lr = np.maximum(y_pred_lr, 0)  # Ensure non-negative
results.append(calculate_metrics(y_test, y_pred_lr, "Linear Regression"))

# 4. Ridge Regression
print("📊 Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_ridge = np.maximum(y_pred_ridge, 0)
results.append(calculate_metrics(y_test, y_pred_ridge, "Ridge Regression"))

# ============================================================================
# MACHINE LEARNING MODELS
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 70)

# 5. Random Forest
print("\n📊 Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_rf = np.maximum(y_pred_rf, 0)
results.append(calculate_metrics(y_test, y_pred_rf, "Random Forest"))

# 6. Gradient Boosting
print("📊 Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    min_samples_split=5,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
y_pred_gb = np.maximum(y_pred_gb, 0)
results.append(calculate_metrics(y_test, y_pred_gb, "Gradient Boosting"))

# 7. XGBoost
print("📊 Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb = np.maximum(y_pred_xgb, 0)
results.append(calculate_metrics(y_test, y_pred_xgb, "XGBoost"))

# ============================================================================
# RESULTS COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON - 1-HOUR AHEAD FORECAST")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv('output/model_comparison_1hr.csv', index=False)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n📊 Analyzing feature importance...")

# Get feature importance from Random Forest
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string(index=False))

# Plot feature importance
fig, ax = plt.subplots(figsize=(12, 8))
top_features = importance_df.head(20)
ax.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue', alpha=0.7)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance')
ax.set_title('Top 20 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Feature importance plot saved")

# ============================================================================
# MULTI-HORIZON FORECASTING
# ============================================================================
print("\n" + "=" * 70)
print("MULTI-HORIZON FORECAST EVALUATION")
print("=" * 70)

horizons = ['15min', '30min', '1hr', '2hr']
all_horizon_results = []

best_model = xgb_model  # Use XGBoost for multi-horizon evaluation

for horizon in horizons:
    target = f'Sales_Target_{horizon}'
    
    # Get target values
    y_test_horizon = test_df[target].values
    y_test_horizon = np.nan_to_num(y_test_horizon, nan=0)
    
    # Predict
    y_pred_horizon = best_model.predict(X_test)
    y_pred_horizon = np.maximum(y_pred_horizon, 0)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_horizon, y_pred_horizon, f"XGBoost ({horizon})")
    metrics['Horizon'] = horizon
    all_horizon_results.append(metrics)
    
    print(f"\n{horizon.upper()} Horizon:")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE (%)']:.2f}%")

horizon_results_df = pd.DataFrame(all_horizon_results)
horizon_results_df.to_csv('output/multi_horizon_results.csv', index=False)

# ============================================================================
# VISUALIZE PREDICTIONS
# ============================================================================
print("\n📊 Creating prediction visualizations...")

# Plot actual vs predicted for test period
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Time series comparison
ax1 = axes[0, 0]
ax1.plot(test_df['Timestamp'].values[:500], y_test[:500], label='Actual', color='steelblue', linewidth=1.5)
ax1.plot(test_df['Timestamp'].values[:500], y_pred_xgb[:500], label='XGBoost Predicted', color='coral', linewidth=1.5, alpha=0.8)
ax1.set_title('Actual vs Predicted Sales (First 500 Test Points)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Sales Count')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# 2. Scatter plot actual vs predicted
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred_xgb, alpha=0.3, s=10, color='purple')
max_val = max(y_test.max(), y_pred_xgb.max())
ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction', linewidth=2)
ax2.set_xlabel('Actual Sales')
ax2.set_ylabel('Predicted Sales')
ax2.set_title('Actual vs Predicted Scatter Plot', fontsize=12, fontweight='bold')
ax2.legend()

# 3. Residuals distribution
ax3 = axes[1, 0]
residuals = y_test - y_pred_xgb
ax3.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
ax3.axvline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Residual (Actual - Predicted)')
ax3.set_ylabel('Frequency')
ax3.set_title('Residuals Distribution', fontsize=12, fontweight='bold')

# 4. Error by hour of day
ax4 = axes[1, 1]
test_df_copy = test_df.copy()
test_df_copy['Error'] = np.abs(y_test - y_pred_xgb)
hourly_error = test_df_copy.groupby('Hour')['Error'].mean()
ax4.bar(hourly_error.index, hourly_error.values, color='coral', alpha=0.7)
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Mean Absolute Error')
ax4.set_title('Prediction Error by Hour of Day', fontsize=12, fontweight='bold')
ax4.set_xticks(range(24))

plt.tight_layout()
plt.savefig('output/figures/08_prediction_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Prediction analysis plot saved")

# Plot horizon comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(horizons))
mae_values = [r['MAE'] for r in all_horizon_results]
rmse_values = [r['RMSE'] for r in all_horizon_results]

width = 0.35
ax.bar([i - width/2 for i in x], mae_values, width, label='MAE', color='steelblue', alpha=0.7)
ax.bar([i + width/2 for i in x], rmse_values, width, label='RMSE', color='coral', alpha=0.7)

ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Error')
ax.set_title('Model Performance Across Forecast Horizons', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.legend()

# Add value labels
for i, (mae, rmse) in enumerate(zip(mae_values, rmse_values)):
    ax.annotate(f'{mae:.1f}', (i - width/2, mae), ha='center', va='bottom', fontsize=10)
    ax.annotate(f'{rmse:.1f}', (i + width/2, rmse), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('output/figures/09_horizon_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Horizon comparison plot saved")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n💾 Saving trained models...")

import joblib
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(gb_model, 'models/gradient_boosting_model.pkl')
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(scaler, 'models/feature_scaler.pkl')

print("✅ Models saved to models/")

# Save feature columns for later use
import json
with open('models/feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)
print("✅ Feature columns saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 70)

print(f"\n📈 BEST MODEL: {results_df.iloc[0]['Model']}")
print(f"   MAE: {results_df.iloc[0]['MAE']:.2f}")
print(f"   RMSE: {results_df.iloc[0]['RMSE']:.2f}")
print(f"   MAPE: {results_df.iloc[0]['MAPE (%)']:.2f}%")

print(f"\n📊 Model Comparison (sorted by RMSE):")
for _, row in results_df.iterrows():
    print(f"   {row['Model']}: MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

print(f"\n⏱️ Forecast Horizon Performance (XGBoost):")
for r in all_horizon_results:
    print(f"   {r['Horizon']}: MAE={r['MAE']:.2f}, RMSE={r['RMSE']:.2f}")

print("\n" + "=" * 70)
print("All outputs saved to output/ and models/")
print("=" * 70)