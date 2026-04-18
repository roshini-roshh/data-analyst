"""
Toronto Island Ferry Ticket Demand - Comprehensive Visualizations
Pattern Analysis for Short-Term Forecasting
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
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Load cleaned data
df = pd.read_csv('data/ferry_tickets_cleaned.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Create output directory for figures
import os
os.makedirs('output/figures', exist_ok=True)

print("Creating comprehensive visualizations...")

# ============================================================================
# FIGURE 1: Time Series Overview
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Sales Count Time Series
ax1 = axes[0]
ax1.plot(df['Timestamp'], df['Sales Count'], color='steelblue', alpha=0.7, linewidth=0.5)
ax1.fill_between(df['Timestamp'], df['Sales Count'], alpha=0.3, color='steelblue')
ax1.set_title('Toronto Island Ferry - Sales Count Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Sales Count (15-min interval)')
ax1.grid(True, alpha=0.3)

# Redemption Count Time Series
ax2 = axes[1]
ax2.plot(df['Timestamp'], df['Redemption Count'], color='coral', alpha=0.7, linewidth=0.5)
ax2.fill_between(df['Timestamp'], df['Redemption Count'], alpha=0.3, color='coral')
ax2.set_title('Toronto Island Ferry - Redemption Count Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Redemption Count (15-min interval)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/figures/01_time_series_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 1: Time Series Overview saved")

# ============================================================================
# FIGURE 2: Hourly Patterns
# ============================================================================
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['DayName'] = df['Timestamp'].dt.day_name()
df['Month'] = df['Timestamp'].dt.month
df['Date'] = df['Timestamp'].dt.date
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Hourly pattern - Sales
hourly_sales = df.groupby('Hour')['Sales Count'].agg(['mean', 'std', 'median'])
ax1 = axes[0, 0]
ax1.bar(hourly_sales.index, hourly_sales['mean'], color='steelblue', alpha=0.7, label='Mean')
ax1.errorbar(hourly_sales.index, hourly_sales['mean'], yerr=hourly_sales['std'], 
             fmt='none', color='darkblue', capsize=3, alpha=0.5)
ax1.plot(hourly_sales.index, hourly_sales['median'], 'ro-', markersize=4, label='Median')
ax1.set_title('Average Sales Count by Hour of Day', fontsize=12, fontweight='bold')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Sales Count')
ax1.legend()
ax1.set_xticks(range(24))

# Hourly pattern - Redemptions
hourly_redemption = df.groupby('Hour')['Redemption Count'].agg(['mean', 'std', 'median'])
ax2 = axes[0, 1]
ax2.bar(hourly_redemption.index, hourly_redemption['mean'], color='coral', alpha=0.7, label='Mean')
ax2.errorbar(hourly_redemption.index, hourly_redemption['mean'], yerr=hourly_redemption['std'], 
             fmt='none', color='darkred', capsize=3, alpha=0.5)
ax2.plot(hourly_redemption.index, hourly_redemption['median'], 'bo-', markersize=4, label='Median')
ax2.set_title('Average Redemption Count by Hour of Day', fontsize=12, fontweight='bold')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Redemption Count')
ax2.legend()
ax2.set_xticks(range(24))

# Day of Week pattern
daily_pattern = df.groupby('DayName')['Sales Count'].mean()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_pattern = daily_pattern.reindex(day_order)
ax3 = axes[1, 0]
colors = ['#3498db' if day not in ['Saturday', 'Sunday'] else '#e74c3c' for day in day_order]
ax3.bar(range(7), daily_pattern.values, color=colors, alpha=0.7)
ax3.set_title('Average Sales Count by Day of Week', fontsize=12, fontweight='bold')
ax3.set_xlabel('Day of Week')
ax3.set_ylabel('Average Sales Count')
ax3.set_xticks(range(7))
ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

# Weekend vs Weekday comparison
weekend_comparison = df.groupby('IsWeekend')[['Sales Count', 'Redemption Count']].mean()
ax4 = axes[1, 1]
x = np.arange(2)
width = 0.35
ax4.bar(x - width/2, [weekend_comparison.loc[0, 'Sales Count'], weekend_comparison.loc[1, 'Sales Count']], 
        width, label='Sales', color='steelblue', alpha=0.7)
ax4.bar(x + width/2, [weekend_comparison.loc[0, 'Redemption Count'], weekend_comparison.loc[1, 'Redemption Count']], 
        width, label='Redemption', color='coral', alpha=0.7)
ax4.set_title('Weekday vs Weekend Average', fontsize=12, fontweight='bold')
ax4.set_xlabel('Day Type')
ax4.set_ylabel('Average Count')
ax4.set_xticks(x)
ax4.set_xticklabels(['Weekday', 'Weekend'])
ax4.legend()

plt.tight_layout()
plt.savefig('output/figures/02_temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 2: Temporal Patterns saved")

# ============================================================================
# FIGURE 3: Distribution Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Sales distribution
ax1 = axes[0, 0]
sales_nonzero = df[df['Sales Count'] > 0]['Sales Count']
ax1.hist(sales_nonzero, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
ax1.axvline(sales_nonzero.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sales_nonzero.mean():.1f}')
ax1.axvline(sales_nonzero.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {sales_nonzero.median():.1f}')
ax1.set_title('Distribution of Sales Count (Non-Zero)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sales Count')
ax1.set_ylabel('Frequency')
ax1.legend()

# Redemption distribution
ax2 = axes[0, 1]
redemption_nonzero = df[df['Redemption Count'] > 0]['Redemption Count']
ax2.hist(redemption_nonzero, bins=50, color='coral', alpha=0.7, edgecolor='white')
ax2.axvline(redemption_nonzero.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {redemption_nonzero.mean():.1f}')
ax2.axvline(redemption_nonzero.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {redemption_nonzero.median():.1f}')
ax2.set_title('Distribution of Redemption Count (Non-Zero)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Redemption Count')
ax2.set_ylabel('Frequency')
ax2.legend()

# Sales vs Redemption scatter
ax3 = axes[1, 0]
ax3.scatter(df['Sales Count'], df['Redemption Count'], alpha=0.3, s=10, color='purple')
ax3.plot([0, df['Sales Count'].max()], [0, df['Sales Count'].max()], 'r--', label='y=x')
ax3.set_title('Sales vs Redemption Count Correlation', fontsize=12, fontweight='bold')
ax3.set_xlabel('Sales Count')
ax3.set_ylabel('Redemption Count')
correlation = df['Sales Count'].corr(df['Redemption Count'])
ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax3.legend()

# Box plot by hour
ax4 = axes[1, 1]
hourly_data = [df[df['Hour'] == h]['Sales Count'].values for h in range(24)]
bp = ax4.boxplot(hourly_data, patch_artist=True, labels=range(24))
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax4.set_title('Sales Count Distribution by Hour', fontsize=12, fontweight='bold')
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Sales Count')

plt.tight_layout()
plt.savefig('output/figures/03_distribution_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 3: Distribution Analysis saved")

# ============================================================================
# FIGURE 4: Heatmap - Hour vs Day of Week
# ============================================================================
pivot_sales = df.pivot_table(values='Sales Count', index='Hour', columns='DayOfWeek', aggfunc='mean')
pivot_sales.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pivot_sales, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=0.5, ax=ax)
ax.set_title('Average Sales Count: Hour vs Day of Week Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Hour of Day')
plt.tight_layout()
plt.savefig('output/figures/04_heatmap_hour_day.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 4: Heatmap saved")

# ============================================================================
# FIGURE 5: Daily Aggregated Trends
# ============================================================================
daily_df = df.groupby('Date').agg({
    'Sales Count': 'sum',
    'Redemption Count': 'sum',
    'DayOfWeek': 'first',
    'IsWeekend': 'first'
}).reset_index()
daily_df['Date'] = pd.to_datetime(daily_df['Date'])

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Daily totals
ax1 = axes[0]
ax1.bar(daily_df['Date'], daily_df['Sales Count'], color='steelblue', alpha=0.7, label='Sales')
ax1.plot(daily_df['Date'], daily_df['Redemption Count'], color='coral', alpha=0.7, linewidth=1.5, label='Redemption')
ax1.set_title('Daily Total Sales and Redemption Counts', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Count')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Rolling average
daily_df['Sales_7d_MA'] = daily_df['Sales Count'].rolling(window=7, min_periods=1).mean()
ax2 = axes[1]
ax2.plot(daily_df['Date'], daily_df['Sales Count'], 'o-', markersize=3, alpha=0.5, color='steelblue', label='Daily Sales')
ax2.plot(daily_df['Date'], daily_df['Sales_7d_MA'], linewidth=2, color='red', label='7-Day Moving Average')
ax2.set_title('Daily Sales with 7-Day Moving Average', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Sales Count')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/figures/05_daily_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 5: Daily Trends saved")

# ============================================================================
# FIGURE 6: Peak Demand Analysis
# ============================================================================
# Identify peak hours (top 10% of sales)
threshold_90 = df['Sales Count'].quantile(0.90)
peak_records = df[df['Sales Count'] >= threshold_90]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Peak hours distribution
ax1 = axes[0, 0]
peak_by_hour = peak_records.groupby('Hour').size()
ax1.bar(peak_by_hour.index, peak_by_hour.values, color='crimson', alpha=0.7)
ax1.set_title(f'Peak Demand Hours (Sales ≥ {threshold_90:.0f})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Number of Peak Records')
ax1.set_xticks(range(24))

# Peak days distribution
ax2 = axes[0, 1]
peak_by_day = peak_records.groupby('DayName').size()
peak_by_day = peak_by_day.reindex(day_order)
ax2.bar(range(7), peak_by_day.values, color='darkorange', alpha=0.7)
ax2.set_title('Peak Demand by Day of Week', fontsize=12, fontweight='bold')
ax2.set_xlabel('Day of Week')
ax2.set_ylabel('Number of Peak Records')
ax2.set_xticks(range(7))
ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

# Time between peaks
ax3 = axes[1, 0]
peak_times = peak_records['Timestamp'].sort_values()
time_diffs = peak_times.diff().dt.total_seconds() / 3600  # in hours
time_diffs = time_diffs.dropna()
ax3.hist(time_diffs[time_diffs < 50], bins=30, color='purple', alpha=0.7, edgecolor='white')
ax3.set_title('Time Between Peak Demand Events', fontsize=12, fontweight='bold')
ax3.set_xlabel('Hours Between Peaks')
ax3.set_ylabel('Frequency')

# Peak intensity over time
ax4 = axes[1, 1]
ax4.scatter(peak_records['Timestamp'], peak_records['Sales Count'], c=peak_records['Hour'], 
            cmap='viridis', alpha=0.6, s=30)
ax4.set_title('Peak Demand Events Over Time', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Sales Count')
cbar = plt.colorbar(ax4.collections[0], ax=ax4)
cbar.set_label('Hour of Day')

plt.tight_layout()
plt.savefig('output/figures/06_peak_demand_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 6: Peak Demand Analysis saved")

# Save enhanced dataset with features
df.to_csv('data/ferry_tickets_features.csv', index=False)
print("\n✅ Enhanced dataset with temporal features saved to data/ferry_tickets_features.csv")

# Print key insights
print("\n" + "=" * 60)
print("KEY INSIGHTS FROM EDA")
print("=" * 60)
print(f"\n1. DATA CHARACTERISTICS:")
print(f"   - Total records: {len(df):,}")
print(f"   - Time span: {df['Timestamp'].min().date()} to {df['Timestamp'].max().date()}")
print(f"   - Records with zero sales: {(df['Sales Count'] == 0).sum()} ({(df['Sales Count'] == 0).sum()/len(df)*100:.1f}%)")

print(f"\n2. TEMPORAL PATTERNS:")
print(f"   - Peak sales hour: {hourly_sales['mean'].idxmax()} (avg: {hourly_sales['mean'].max():.1f})")
print(f"   - Peak redemption hour: {hourly_redemption['mean'].idxmax()} (avg: {hourly_redemption['mean'].max():.1f})")
print(f"   - Busiest day: {daily_pattern.idxmax()} (avg: {daily_pattern.max():.1f})")

print(f"\n3. DEMAND VARIABILITY:")
print(f"   - Sales coefficient of variation: {(df['Sales Count'].std() / df['Sales Count'].mean())*100:.1f}%")
print(f"   - Max sales in single interval: {df['Sales Count'].max()}")
print(f"   - 90th percentile threshold: {threshold_90:.0f}")

print(f"\n4. WEEKEND VS WEEKDAY:")
print(f"   - Weekday avg sales: {weekend_comparison.loc[0, 'Sales Count']:.1f}")
print(f"   - Weekend avg sales: {weekend_comparison.loc[1, 'Sales Count']:.1f}")
weekend_lift = (weekend_comparison.loc[1, 'Sales Count'] / weekend_comparison.loc[0, 'Sales Count'] - 1) * 100
print(f"   - Weekend lift: {weekend_lift:+.1f}%")

print(f"\n5. CORRELATION:")
print(f"   - Sales-Redemption correlation: {correlation:.3f}")

print("\n" + "=" * 60)
print("All visualizations saved to output/figures/")
print("=" * 60)