"""
Toronto Island Ferry Ticket Demand - Exploratory Data Analysis
Short-Term Forecasting & Predictive Decision Support System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the dataset
print("=" * 60)
print("TORONTO ISLAND FERRY TICKET COUNTS - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv('data/ferry_tickets.csv')

# Basic info
print("\n📊 DATASET OVERVIEW")
print("-" * 40)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort by timestamp (oldest first)
df = df.sort_values('Timestamp').reset_index(drop=True)

print("\n📅 TIME RANGE")
print("-" * 40)
print(f"Start Date: {df['Timestamp'].min()}")
print(f"End Date: {df['Timestamp'].max()}")
time_span = df['Timestamp'].max() - df['Timestamp'].min()
print(f"Time Span: {time_span}")
print(f"Total Days: {time_span.days}")

# Check for missing values
print("\n🔍 MISSING VALUES")
print("-" * 40)
print(df.isnull().sum())

# Statistical summary
print("\n📈 STATISTICAL SUMMARY")
print("-" * 40)
print(df[['Sales Count', 'Redemption Count']].describe())

# Check for any zero or negative values
print("\n⚠️ DATA QUALITY CHECKS")
print("-" * 40)
print(f"Zero Sales Count: {(df['Sales Count'] == 0).sum()} records")
print(f"Zero Redemption Count: {(df['Redemption Count'] == 0).sum()} records")
print(f"Negative Sales Count: {(df['Sales Count'] < 0).sum()} records")
print(f"Negative Redemption Count: {(df['Redemption Count'] < 0).sum()} records")

# Time interval analysis
df['time_diff'] = df['Timestamp'].diff()
intervals = df['time_diff'].dropna().unique()
print(f"\nUnique Time Intervals: {len(intervals)}")
print(f"Expected interval: 15 minutes")

# Save cleaned data for further processing
df.to_csv('data/ferry_tickets_cleaned.csv', index=False)
print("\n✅ Cleaned data saved to data/ferry_tickets_cleaned.csv")

# Create summary statistics file
summary = {
    'total_records': len(df),
    'date_start': str(df['Timestamp'].min()),
    'date_end': str(df['Timestamp'].max()),
    'total_days': time_span.days,
    'sales_mean': df['Sales Count'].mean(),
    'sales_std': df['Sales Count'].std(),
    'sales_max': df['Sales Count'].max(),
    'sales_min': df['Sales Count'].min(),
    'redemption_mean': df['Redemption Count'].mean(),
    'redemption_std': df['Redemption Count'].std(),
    'redemption_max': df['Redemption Count'].max(),
    'redemption_min': df['Redemption Count'].min()
}

print("\n📋 SUMMARY STATISTICS")
print("-" * 40)
for key, value in summary.items():
    print(f"{key}: {value}")