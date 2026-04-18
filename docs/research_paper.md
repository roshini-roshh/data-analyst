# Short-Term Ferry Ticket Demand Forecasting & Predictive Decision Support System

## Toronto Island Park Ferry Operations

---

## Executive Summary

This research presents a comprehensive machine learning approach to short-term demand forecasting for Toronto Island Park ferry operations. By analyzing over 9,500 records of 15-minute interval ticket data spanning 150 days, we developed and compared multiple forecasting models to predict demand for 15-minute to 2-hour horizons. Our findings demonstrate that Moving Average models provide robust baseline predictions, while ensemble methods like Random Forest and XGBoost offer competitive performance for operational planning. The implementation of a Streamlit-based decision support dashboard enables real-time operational intelligence for ferry scheduling, staff readiness, and crowd management.

---

## 1. Introduction

### 1.1 Background

Toronto Island Park is a major recreational destination accessible primarily by ferry services operated by the City of Toronto Parks, Forestry & Recreation department. Ferry operations require anticipatory decision-making to ensure efficient service delivery, adequate staffing, and passenger safety during peak demand periods. Traditional reactive approaches to operations management often result in suboptimal resource allocation and potential service disruptions.

### 1.2 Problem Statement

Despite having over a decade of high-frequency ticket data, ferry operations currently lack short-term demand forecasts, predictive visibility into upcoming congestion, and quantitative uncertainty estimates for planning decisions. Operational responses are therefore reactive rather than predictive, increasing the risk of congestion, delays, and service inefficiencies.

### 1.3 Objectives

**Primary Objectives:**
- Forecast short-term ferry ticket sales and redemptions
- Predict demand for upcoming 15-minute to 2-hour windows
- Compare statistical and machine-learning forecasting approaches

**Secondary Objectives:**
- Quantify prediction uncertainty
- Support proactive operational planning
- Demonstrate real-world ML deployment via Streamlit

---

## 2. Data Description and Methodology

### 2.1 Dataset Overview

The dataset was obtained from the City of Toronto Open Data Portal and contains the following structure:

| Column | Description |
|--------|-------------|
| `_id` | Unique row identifier |
| `Timestamp` | 15-minute interval end time |
| `Sales Count` | Tickets sold in interval |
| `Redemption Count` | Tickets redeemed in interval |

**Dataset Statistics:**
- Total records: 9,519 (original) → 14,436 (after interpolation for missing intervals)
- Time span: November 4, 2025 to April 3, 2026 (150 days)
- Sampling frequency: 15-minute intervals

### 2.2 Data Quality Assessment

Initial data quality analysis revealed:
- **Missing intervals**: 4,917 missing 15-minute intervals identified
- **Zero sales records**: 1,451 records (15.2% of data) with zero sales
- **No negative values**: Data integrity confirmed
- **High variability**: Coefficient of variation of 214.9% for sales

### 2.3 Methodology

Our forecasting methodology followed a systematic approach:

**Phase 1: Time-Series Preparation**
- Converted timestamps to datetime index
- Ensured strict chronological ordering
- Handled missing 15-minute intervals via linear interpolation

**Phase 2: Feature Engineering**
- Created lag features (t-1, t-2, t-4, t-8, up to t-96 for 24-hour lookback)
- Generated rolling statistics (mean, std, max, min) over 1-hour, 2-hour, 4-hour, 8-hour, and 24-hour windows
- Added temporal encodings (hour, day of week, month, weekend indicator)
- Implemented cyclical encoding using sine/cosine transformations

**Phase 3: Train-Test Strategy**
- Time-based split (80% training, 20% testing)
- No random shuffling to preserve temporal dependencies
- Multiple forecast horizons evaluated: 15 minutes, 30 minutes, 1 hour, 2 hours

---

## 3. Exploratory Data Analysis

### 3.1 Temporal Patterns

**Hourly Patterns:**
- Peak sales hour: 15:00 (3 PM) with average of 23.0 tickets per interval
- Peak redemption hour: 15:00 (3 PM) with average of 23.6 tickets per interval
- Low activity periods: Early morning (0:00-6:00) and late evening (20:00-24:00)

**Day of Week Patterns:**
- Busiest day: Saturday with average of 14.5 tickets per interval
- Lowest activity: Tuesday-Wednesday with average of ~10 tickets per interval
- Weekend lift: +18.5% higher demand compared to weekdays

### 3.2 Demand Distribution Analysis

The distribution of sales counts exhibits significant right-skewness:
- Mean sales: 11.6 tickets per interval
- Median sales: 5 tickets per interval
- Maximum sales: 1,000 tickets in a single interval
- 90th percentile threshold: 27 tickets

### 3.3 Correlation Analysis

Sales and redemption counts show strong positive correlation (r = 0.805), indicating that ticket purchasing and redemption activities are closely linked, likely driven by the same passenger flow patterns.

### 3.4 Key Insights

1. **Strong seasonality**: Clear hourly and daily patterns exist, with afternoon hours and weekends showing elevated demand
2. **High variability**: The demand exhibits significant fluctuation, requiring robust forecasting methods
3. **Data sparsity**: A significant proportion of intervals have zero sales, likely corresponding to off-hours
4. **Weekend effect**: Weekend demand is consistently higher than weekday demand

---

## 4. Model Development

### 4.1 Baseline Models

**Naive Forecast:**
- Prediction: Use current value as prediction
- Purpose: Establish minimum performance benchmark

**Moving Average:**
- Prediction: Average of last 4 intervals (1 hour)
- Purpose: Simple smoothing-based forecasting

**Linear Regression:**
- Features: All engineered features
- Purpose: Linear relationship modeling

**Ridge Regression:**
- Features: All engineered features with L2 regularization
- Purpose: Regularized linear model to prevent overfitting

### 4.2 Machine Learning Models

**Random Forest Regressor:**
- Configuration: 100 estimators, max depth 15
- Strengths: Handles non-linear relationships, feature importance

**Gradient Boosting Regressor:**
- Configuration: 100 estimators, max depth 8, learning rate 0.1
- Strengths: Sequential error correction, strong predictive power

**XGBoost:**
- Configuration: 100 estimators, max depth 8, learning rate 0.1
- Strengths: Regularization, handling missing values, competitive performance

### 4.3 Feature Importance Analysis

The Random Forest model identified the following as the most important features for demand prediction:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Hour_cos | 0.098 |
| 2 | Sales_Rolling_Mean_96 | 0.069 |
| 3 | Sales_Rolling_Mean_16 | 0.068 |
| 4 | Sales_Lag_4 | 0.060 |
| 5 | Sales_Rolling_Min_4 | 0.047 |

The cyclical encoding of hour (Hour_cos) emerges as the most important feature, confirming the strong hourly seasonality in demand patterns.

---

## 5. Results and Evaluation

### 5.1 Model Performance Comparison (1-Hour Forecast)

| Model | MAE | RMSE | MAPE (%) | Within ±10 (%) |
|-------|-----|------|----------|----------------|
| Moving Average (1hr) | 8.04 | 18.64 | 152.90 | 78.87 |
| Ridge Regression | 9.96 | 18.95 | 293.97 | 66.89 |
| Linear Regression | 10.35 | 19.22 | 293.47 | 67.78 |
| Naive (t=t-1) | 8.54 | 22.15 | 126.87 | 78.15 |
| Random Forest | 8.96 | 22.32 | 163.52 | 75.75 |
| Gradient Boosting | 12.32 | 25.09 | 277.13 | 67.10 |
| XGBoost | 11.23 | 26.75 | 203.99 | 71.49 |

### 5.2 Multi-Horizon Performance (XGBoost)

| Horizon | MAE | RMSE | MAPE (%) |
|---------|-----|------|----------|
| 15min | 11.34 | 26.85 | 193.96 |
| 30min | 11.28 | 26.97 | 192.41 |
| 1hr | 11.23 | 26.75 | 203.99 |
| 2hr | 11.51 | 27.59 | 234.61 |

### 5.3 Key Findings

1. **Moving Average Performs Best**: The simple Moving Average model achieves the lowest RMSE (18.64) for 1-hour ahead forecasting, demonstrating that temporal smoothing is effective for this dataset.

2. **Consistent Error Across Horizons**: Prediction error remains relatively stable across different forecast horizons (15min to 2hr), suggesting that short-term demand is influenced by recent patterns regardless of horizon length.

3. **Linear Models Competitive**: Ridge and Linear Regression models show competitive performance, indicating that linear relationships captured by engineered features are informative.

4. **Feature Engineering Impact**: The feature importance analysis confirms that cyclical temporal features and rolling statistics are most predictive, validating our feature engineering approach.

---

## 6. Streamlit Dashboard Implementation

### 6.1 Dashboard Features

The implemented Streamlit dashboard provides:

**Dashboard Overview:**
- Total sales and redemption metrics
- Time series visualization
- Hourly demand heatmap

**Demand Analysis:**
- Hourly pattern analysis with error bars
- Day of week comparison
- Sales distribution visualization

**Forecasting Module:**
- Interactive model selection
- Forecast horizon selector (15min to 2hr)
- Real-time predictions with confidence intervals
- Demand alerts for high-demand periods

**Model Performance:**
- Model comparison charts
- Multi-horizon performance visualization
- Feature importance display

**KPI Metrics:**
- Forecast accuracy metrics
- Operational KPIs
- Actionable recommendations

### 6.2 Technical Implementation

- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts
- **Models**: Joblib-serialized scikit-learn models
- **Data Processing**: Pandas and NumPy

---

## 7. Recommendations

### 7.1 Operational Recommendations

Based on our analysis, we recommend the following operational improvements:

1. **Staffing Optimization**: Increase staff capacity during peak hours (14:00-16:00) and on weekends, particularly Saturdays which show 18.5% higher demand than weekdays.

2. **Proactive Scheduling**: Use the Moving Average model for short-term demand predictions to schedule additional ferry services during anticipated high-demand periods.

3. **Maintenance Planning**: Schedule routine maintenance during low-demand hours (early morning 6:00-9:00 and late evening after 19:00) to minimize service disruption.

4. **Real-time Monitoring**: Deploy the Streamlit dashboard for real-time demand monitoring and enable proactive crowd management.

### 7.2 Model Improvement Recommendations

1. **External Factors**: Incorporate weather data, holidays, and special events as additional features to improve prediction accuracy.

2. **Ensemble Methods**: Develop ensemble models combining Moving Average with machine learning approaches for improved robustness.

3. **Uncertainty Quantification**: Implement bootstrap prediction intervals to provide operational planners with confidence bounds.

4. **Real-time Updates**: Deploy a model retraining pipeline to incorporate new data and adapt to changing demand patterns.

---

## 8. Conclusion

This project successfully developed a short-term demand forecasting system for Toronto Island Park ferry operations. Our analysis demonstrates that:

- Moving Average models provide the most accurate short-term predictions (RMSE = 18.64 for 1-hour forecasts)
- Temporal patterns (hour of day, day of week) are strong predictors of demand
- Weekend demand is significantly higher than weekday demand (+18.5%)
- The implemented Streamlit dashboard enables real-time operational intelligence

The transition from reactive to predictive operations management will enable ferry operators to proactively schedule resources, manage crowds effectively, and improve overall service efficiency.

---

## References

1. City of Toronto Open Data Portal: Toronto Island Ferry Ticket Counts Dataset
2. Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd edition
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system

---

## Appendix

### A. Data Sources

- Primary Dataset: Toronto Island Ferry Ticket Counts from City of Toronto Open Data Portal
- Dataset URL: https://open.toronto.ca/dataset/toronto-island-ferry-ticket-counts/

### B. Code Repository Structure

```
/workspace/
├── data/
│   ├── ferry_tickets.csv
│   ├── ferry_tickets_cleaned.csv
│   ├── ferry_tickets_features.csv
│   └── ferry_tickets_engineered.csv
├── models/
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── xgboost_model.pkl
│   └── feature_scaler.pkl
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_visualizations.py
│   ├── 03_feature_engineering.py
│   └── 04_model_training.py
├── output/
│   └── figures/
├── docs/
│   └── research_paper.md
├── app.py
└── todo.md
```

### C. Feature Engineering Summary

Total features created: 49 (excluding target variables)

| Category | Count | Examples |
|----------|-------|----------|
| Temporal | 13 | Hour, DayOfWeek, Month, IsWeekend, Cyclical encodings |
| Lag | 9 | Sales_Lag_1, Sales_Lag_4, Sales_Lag_96 |
| Rolling | 20 | Sales_Rolling_Mean_4, Sales_Rolling_Std_8 |
| Difference | 3 | Sales_Diff_1, Sales_PctChange_1, Sales_Diff_24h |
| Aggregated | 4 | Sales_Previous_Day_Total, Sales_Hourly_Avg |