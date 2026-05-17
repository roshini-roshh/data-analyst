🚢 Short-Term Ferry Ticket Demand Forecasting
Predictive Decision Support System for Transportation Analytics
📌 Project Overview

This project presents a Short-Term Ferry Ticket Demand Forecasting system designed to support operational decision-making for ferry transportation services.

Using over a decade of 15-minute interval ticket transaction data, the system applies advanced time-series forecasting and machine learning models to predict demand across multiple time horizons (15 min, 30 min, 1 hr, 2 hr).

The final solution is deployed as an interactive Streamlit dashboard for real-time operational insights.

🎯 Objectives
Predict short-term ferry ticket demand accurately
Support proactive ferry scheduling and crowd management
Compare multiple forecasting approaches
Provide uncertainty-aware predictions for decision-making
📊 Key Features
⏱ 15-min interval demand forecasting
📈 Multi-horizon prediction (15 min → 2 hours)
🤖 Multiple model comparison:
Random Forest
Gradient Boosting
XGBoost
SARIMA
Facebook Prophet
📉 Error analysis (MAE, RMSE, MAPE)
📊 Prediction confidence intervals
🧠 Feature engineering (lags, rolling stats, calendar features)
📡 Interactive Streamlit dashboard
🧠 Methodology
1. Data Processing
Time-indexed 15-minute interval data
Missing value handling (interpolation + masking)
Strict chronological train-test split (no leakage)
2. Feature Engineering
Lag features (t-1, t-2, t-4, t-8)
Rolling statistics (mean, std, max)
Calendar features (hour, day, month, weekend)
Cross-series features (sales & redemption relationship)
3. Models Used
Baseline: Naïve, Moving Average, Linear Regression
ML Models: Random Forest, Gradient Boosting, XGBoost
Time-Series: SARIMA, Facebook Prophet
📈 Key Results
✅ Gradient Boosting achieved best short-term accuracy (~9% MAPE)
📊 Prophet performed best for long-term forecasting
📉 ML models significantly outperformed baseline methods
⚡ Strong correlation between lag features and demand patterns
🖥 Dashboard Features
Real-time demand forecasting
Model selection interface
Confidence interval visualization
Risk-level indicators (Low / Medium / High demand)
Time horizon selection
🛠 Tech Stack
Python 🐍
Pandas / NumPy
Scikit-learn
XGBoost
Statsmodels (SARIMA)
Facebook Prophet
Streamlit 📊
Matplotlib / Seaborn

📁 Project Structure
├── dashboard.py
├── eda_analysis.py
├── generate_data.py
├── research_paper.md
├── executive_summary.md
├── metrics.json
├── charts/
├── outputs/

📌 Key Insights
Demand shows strong daily + weekly seasonality
Lag features are highly predictive for short-term forecasting
ML models outperform statistical models in short horizons
Forecast uncertainty increases significantly beyond 1 hour
🚧 Limitations
Weather and event data not included
No deep learning models implemented (future improvement)
Sales and redemption modeled separately (not joint forecasting)
🚀 Future Improvements
Add weather + event-based forecasting
Implement LSTM / Temporal CNN models
Deploy cloud-based real-time API system
Improve multivariate forecasting (sales + redemption joint model)
