"""
Toronto Island Ferry Ticket Demand - Feature Engineering for Forecasting
Creates lag features, rolling statistics, and temporal encodings
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FEATURE ENGINEERING FOR SHORT-TERM DEMAND FORECASTING")
print("=" * 60)

# Load the data with temporal features
df = pd.read_csv('data/ferry_tickets_features.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort by timestamp
df = df.sort_values('Timestamp').reset_index(drop=True)

print(f"\n📊 Original dataset shape: {df.shape}")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# ============================================================================
# 1. CREATE PROPER TIME SERIES INDEX
# ============================================================================
print("\n🔧 Creating proper time series structure...")

# Set timestamp as index
df = df.set_index('Timestamp')

# Check for gaps in the 15-minute intervals
expected_intervals = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')
missing_intervals = expected_intervals.difference(df.index)
print(f"Expected intervals: {len(expected_intervals)}")
print(f"Actual records: {len(df)}")
print(f"Missing intervals: {len(missing_intervals)}")

if len(missing_intervals) > 0:
    print(f"   First few missing: {missing_intervals[:5].tolist()}")
    
    # Reindex to fill missing intervals with NaN
    df = df.reindex(expected_intervals)
    print(f"   After reindexing: {len(df)} records")
    
    # Interpolate missing values
    df['Sales Count'] = df['Sales Count'].interpolate(method='linear')
    df['Redemption Count'] = df['Redemption Count'].interpolate(method='linear')
    df['_id'] = df['_id'].interpolate(method='linear')
    
    # Fill any remaining NaN at the start
    df = df.bfill()
    
    df = df.reset_index()
    df = df.rename(columns={'index': 'Timestamp'})

print(f"\n✓ Time series structure created: {df.shape}")

# ============================================================================
# 2. TEMPORAL FEATURES
# ============================================================================
print("\n🔧 Creating temporal features...")

df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['DayOfYear'] = df['Timestamp'].dt.dayofyear
df['Month'] = df['Timestamp'].dt.month
df['Week'] = df['Timestamp'].dt.isocalendar().week.astype(int)
df['Quarter'] = df['Timestamp'].dt.quarter
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Time of day categories
def time_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

df['TimeOfDay'] = df['Hour'].apply(time_of_day)

# Cyclical encoding for temporal features (helps ML models understand cyclicity)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

print("✓ Temporal features created: Hour, DayOfWeek, Month, IsWeekend, Cyclical encodings")

# ============================================================================
# 3. LAG FEATURES
# ============================================================================
print("\n🔧 Creating lag features...")

# Create lag features for different horizons
# For a 15-minute interval, we create lags for: t-1, t-2, t-4, t-8 intervals
# This corresponds to: 15min, 30min, 1hr, 2hr ago

lag_intervals = [1, 2, 3, 4, 8, 16, 32, 48, 96]  # 15min, 30min, 45min, 1hr, 2hr, 4hr, 8hr, 12hr, 24hr

for lag in lag_intervals:
    df[f'Sales_Lag_{lag}'] = df['Sales Count'].shift(lag)
    df[f'Redemption_Lag_{lag}'] = df['Redemption Count'].shift(lag)

print(f"✓ Lag features created for intervals: {lag_intervals}")

# ============================================================================
# 4. ROLLING STATISTICS
# ============================================================================
print("\n🔧 Creating rolling statistics...")

# Rolling windows in terms of 15-minute intervals
windows = [4, 8, 16, 32, 96]  # 1hr, 2hr, 4hr, 8hr, 24hr

for window in windows:
    # Rolling mean
    df[f'Sales_Rolling_Mean_{window}'] = df['Sales Count'].shift(1).rolling(window=window).mean()
    df[f'Redemption_Rolling_Mean_{window}'] = df['Redemption Count'].shift(1).rolling(window=window).mean()
    
    # Rolling std
    df[f'Sales_Rolling_Std_{window}'] = df['Sales Count'].shift(1).rolling(window=window).std()
    df[f'Redemption_Rolling_Std_{window}'] = df['Redemption Count'].shift(1).rolling(window=window).std()
    
    # Rolling max
    df[f'Sales_Rolling_Max_{window}'] = df['Sales Count'].shift(1).rolling(window=window).max()
    df[f'Redemption_Rolling_Max_{window}'] = df['Redemption Count'].shift(1).rolling(window=window).max()
    
    # Rolling min
    df[f'Sales_Rolling_Min_{window}'] = df['Sales Count'].shift(1).rolling(window=window).min()
    df[f'Redemption_Rolling_Min_{window}'] = df['Redemption Count'].shift(1).rolling(window=window).min()

print(f"✓ Rolling statistics created for windows: {windows}")

# ============================================================================
# 5. DIFFERENCE FEATURES (Rate of Change)
# ============================================================================
print("\n🔧 Creating difference features...")

# First differences
df['Sales_Diff_1'] = df['Sales Count'].diff(1)
df['Redemption_Diff_1'] = df['Redemption Count'].diff(1)

# Percentage changes
df['Sales_PctChange_1'] = df['Sales Count'].pct_change(1)
df['Redemption_PctChange_1'] = df['Redemption Count'].pct_change(1)

# Difference from same time yesterday (96 intervals = 24 hours)
df['Sales_Diff_24h'] = df['Sales Count'] - df['Sales Count'].shift(96)
df['Redemption_Diff_24h'] = df['Redemption Count'] - df['Redemption Count'].shift(96)

print("✓ Difference features created")

# ============================================================================
# 6. AGGREGATED FEATURES
# ============================================================================
print("\n🔧 Creating aggregated historical features...")

# Previous day total
df['Sales_Previous_Day_Total'] = df['Sales Count'].shift(96).rolling(window=96).sum()

# Previous week same time
df['Sales_Same_Time_Last_Week'] = df['Sales Count'].shift(96 * 7)

# Hourly average for this hour (historical)
df['Sales_Hourly_Avg'] = df.groupby('Hour')['Sales Count'].transform('mean')
df['Redemption_Hourly_Avg'] = df.groupby('Hour')['Redemption Count'].transform('mean')

# Day of week average
df['Sales_DayOfWeek_Avg'] = df.groupby('DayOfWeek')['Sales Count'].transform('mean')

print("✓ Aggregated features created")

# ============================================================================
# 7. TARGET VARIABLES FOR DIFFERENT FORECAST HORIZONS
# ============================================================================
print("\n🔧 Creating target variables for different forecast horizons...")

# Forecast horizons: 15min, 30min, 1hr, 2hr ahead
horizons = {
    '15min': 1,
    '30min': 2,
    '1hr': 4,
    '2hr': 8
}

for horizon_name, horizon_steps in horizons.items():
    df[f'Sales_Target_{horizon_name}'] = df['Sales Count'].shift(-horizon_steps)
    df[f'Redemption_Target_{horizon_name}'] = df['Redemption Count'].shift(-horizon_steps)

print(f"✓ Target variables created for horizons: {list(horizons.keys())}")

# ============================================================================
# 8. CLEAN UP AND SAVE
# ============================================================================
print("\n🔧 Cleaning up dataset...")

# Remove rows with NaN values (due to lag features)
initial_rows = len(df)
df_clean = df.dropna()
final_rows = len(df_clean)

print(f"Rows removed due to NaN: {initial_rows - final_rows}")
print(f"Final dataset shape: {df_clean.shape}")

# Save the feature-engineered dataset
df_clean.to_csv('data/ferry_tickets_engineered.csv', index=False)
print("\n✅ Feature-engineered dataset saved to data/ferry_tickets_engineered.csv")

# Print feature summary
print("\n" + "=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)

feature_categories = {
    'Temporal': [c for c in df_clean.columns if any(x in c for x in ['Hour', 'Day', 'Month', 'Week', 'Quarter', 'IsWeekend', 'TimeOfDay', 'sin', 'cos'])],
    'Lag': [c for c in df_clean.columns if 'Lag' in c],
    'Rolling': [c for c in df_clean.columns if 'Rolling' in c],
    'Difference': [c for c in df_clean.columns if any(x in c for x in ['Diff', 'PctChange'])],
    'Aggregated': [c for c in df_clean.columns if any(x in c for x in ['Previous', 'Same_Time', 'Hourly_Avg', 'DayOfWeek_Avg'])],
    'Target': [c for c in df_clean.columns if 'Target' in c]
}

for category, features in feature_categories.items():
    print(f"\n{category} Features ({len(features)}):")
    for f in features[:5]:  # Show first 5
        print(f"   - {f}")
    if len(features) > 5:
        print(f"   ... and {len(features) - 5} more")

print("\n" + "=" * 60)
print(f"Total features: {len(df_clean.columns)}")
print(f"Total records: {len(df_clean)}")
print("=" * 60)

# Create a feature list for model training
feature_list = [c for c in df_clean.columns if c not in ['_id', 'Timestamp', 'Sales Count', 'Redemption Count', 'TimeOfDay', 'Date']]
feature_list = [c for c in feature_list if not c.startswith('Target')]

print(f"\nFeatures available for model training: {len(feature_list)}")
print(f"Target horizons: {list(horizons.keys())}")

# Save feature list
pd.DataFrame({'feature': feature_list}).to_csv('data/feature_list.csv', index=False)
print("✅ Feature list saved to data/feature_list.csv")