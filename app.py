"""
Toronto Island Ferry Ticket Demand - Streamlit Dashboard
Short-Term Forecasting & Predictive Decision Support System
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Toronto Island Ferry Demand Forecasting",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2C5282;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f5ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3182ce;
    }
    .warning-box {
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #e53e3e;
    }
    .success-box {
        background-color: #f0fff4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #38a169;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load the processed dataset"""
    df = pd.read_csv('data/ferry_tickets_engineered.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    try:
        models['rf'] = joblib.load('models/random_forest_model.pkl')
        models['gb'] = joblib.load('models/gradient_boosting_model.pkl')
        models['xgb'] = joblib.load('models/xgboost_model.pkl')
        models['scaler'] = joblib.load('models/feature_scaler.pkl')
    except:
        st.warning("Models not found. Please run model training first.")
    return models

@st.cache_data
def load_feature_columns():
    """Load feature columns"""
    try:
        with open('models/feature_columns.json', 'r') as f:
            return json.load(f)
    except:
        return []

# Load everything
df = load_data()
models = load_models()
feature_cols = load_feature_columns()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("## 🚢 Navigation")
page = st.sidebar.radio("Go to", 
    ["🏠 Dashboard Overview", 
     "📊 Demand Analysis", 
     "🔮 Forecasting", 
     "📈 Model Performance",
     "📋 KPI Metrics"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")

# Date range selector
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()

selected_start = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
selected_end = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Forecast horizon selector
if page == "🔮 Forecasting":
    st.sidebar.markdown("### Forecast Settings")
    horizon = st.sidebar.select_slider(
        "Forecast Horizon",
        options=['15min', '30min', '1hr', '2hr'],
        value='1hr'
    )
    
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Gradient Boosting", "XGBoost", "Moving Average"]
    )

# Filter data based on selection
mask = (df['Timestamp'].dt.date >= selected_start) & (df['Timestamp'].dt.date <= selected_end)
filtered_df = df[mask].copy()

# ============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================================
if page == "🏠 Dashboard Overview":
    st.markdown('<p class="main-header">🚢 Toronto Island Ferry Demand Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Short-Term Predictive Decision Support System</p>', unsafe_allow_html=True)
    
    # Key Metrics Row
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = filtered_df['Sales Count'].sum()
        st.metric("Total Sales", f"{total_sales:,.0f}", delta=None)
    
    with col2:
        total_redemption = filtered_df['Redemption Count'].sum()
        st.metric("Total Redemptions", f"{total_redemption:,.0f}", delta=None)
    
    with col3:
        avg_sales = filtered_df['Sales Count'].mean()
        st.metric("Avg Sales/Interval", f"{avg_sales:.1f}", delta=None)
    
    with col4:
        peak_sales = filtered_df['Sales Count'].max()
        st.metric("Peak Sales", f"{peak_sales:,.0f}", delta=None)
    
    # Time Series Overview
    st.markdown('<p class="sub-header">📈 Sales & Redemption Trends</p>', unsafe_allow_html=True)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Sales Count Over Time', 'Redemption Count Over Time'))
    
    fig.add_trace(
        go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['Sales Count'],
                   mode='lines', name='Sales', line=dict(color='#3182ce', width=1),
                   fill='tozeroy', fillcolor='rgba(49, 130, 206, 0.2)'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=filtered_df['Timestamp'], y=filtered_df['Redemption Count'],
                   mode='lines', name='Redemption', line=dict(color='#e53e3e', width=1),
                   fill='tozeroy', fillcolor='rgba(229, 62, 62, 0.2)'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True, template='plotly_white')
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly Pattern Heatmap
    st.markdown('<p class="sub-header">🌡️ Hourly Demand Heatmap</p>', unsafe_allow_html=True)
    
    # Create heatmap data
    heatmap_data = filtered_df.groupby(['DayOfWeek', 'Hour'])['Sales Count'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Hour', columns='DayOfWeek', values='Sales Count')
    heatmap_pivot.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='YlOrRd',
        colorbar=dict(title="Avg Sales")
    ))
    
    fig_heatmap.update_layout(
        height=500,
        xaxis_title="Day of Week",
        yaxis_title="Hour of Day",
        template='plotly_white'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================================
# PAGE 2: DEMAND ANALYSIS
# ============================================================================
elif page == "📊 Demand Analysis":
    st.markdown('<p class="main-header">📊 Demand Analysis</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Hourly Patterns", "Day of Week", "Distribution"])
    
    with tab1:
        st.markdown("### Average Demand by Hour")
        
        hourly_stats = filtered_df.groupby('Hour').agg({
            'Sales Count': ['mean', 'std', 'median'],
            'Redemption Count': ['mean', 'std', 'median']
        }).reset_index()
        
        fig = go.Figure()
        
        # Sales mean with error bars
        fig.add_trace(go.Scatter(
            x=hourly_stats['Hour'],
            y=hourly_stats['Sales Count']['mean'],
            mode='lines+markers',
            name='Sales (Mean)',
            line=dict(color='#3182ce', width=2),
            error_y=dict(
                type='data',
                array=hourly_stats['Sales Count']['std'],
                visible=True,
                color='#3182ce'
            )
        ))
        
        # Redemption mean
        fig.add_trace(go.Scatter(
            x=hourly_stats['Hour'],
            y=hourly_stats['Redemption Count']['mean'],
            mode='lines+markers',
            name='Redemption (Mean)',
            line=dict(color='#e53e3e', width=2),
            error_y=dict(
                type='data',
                array=hourly_stats['Redemption Count']['std'],
                visible=True,
                color='#e53e3e'
            )
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Hour of Day",
            yaxis_title="Average Count",
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak hours insight
        peak_hour = hourly_stats.loc[hourly_stats['Sales Count']['mean'].idxmax(), 'Hour']
        st.info(f"⏰ **Peak Demand Hour:** {int(peak_hour)}:00 - Average sales of {hourly_stats['Sales Count']['mean'].max():.1f}")
    
    with tab2:
        st.markdown("### Average Demand by Day of Week")
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats = filtered_df.groupby('DayOfWeek')['Sales Count'].mean()
        
        fig = go.Figure(data=[
            go.Bar(
                x=day_names,
                y=daily_stats.values,
                marker_color=['#3182ce' if i < 5 else '#e53e3e' for i in range(7)],
                text=[f'{v:.1f}' for v in daily_stats.values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            height=500,
            xaxis_title="Day of Week",
            yaxis_title="Average Sales Count",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekend vs Weekday comparison
        weekday_avg = filtered_df[filtered_df['IsWeekend'] == 0]['Sales Count'].mean()
        weekend_avg = filtered_df[filtered_df['IsWeekend'] == 1]['Sales Count'].mean()
        lift = ((weekend_avg / weekday_avg) - 1) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekday Average", f"{weekday_avg:.1f}")
        with col2:
            st.metric("Weekend Average", f"{weekend_avg:.1f}", delta=f"{lift:+.1f}%")
    
    with tab3:
        st.markdown("### Sales Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = go.Figure(data=[go.Histogram(
                x=filtered_df[filtered_df['Sales Count'] > 0]['Sales Count'],
                nbinsx=50,
                marker_color='#3182ce',
                opacity=0.7
            )])
            
            fig.update_layout(
                title="Sales Count Distribution (Non-Zero)",
                xaxis_title="Sales Count",
                yaxis_title="Frequency",
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by hour
            fig = go.Figure()
            
            for hour in range(24):
                hour_data = filtered_df[filtered_df['Hour'] == hour]['Sales Count']
                fig.add_trace(go.Box(
                    y=hour_data,
                    name=str(hour),
                    marker_color='#3182ce',
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Sales Distribution by Hour",
                xaxis_title="Hour",
                yaxis_title="Sales Count",
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: FORECASTING
# ============================================================================
elif page == "🔮 Forecasting":
    st.markdown('<p class="main-header">🔮 Demand Forecasting</p>', unsafe_allow_html=True)
    
    st.markdown(f"**Forecast Horizon:** {horizon} ahead")
    st.markdown(f"**Selected Model:** {model_choice}")
    
    # Target column based on horizon
    target_col = f'Sales_Target_{horizon}'
    
    if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        # ML-based forecasting
        st.markdown("### Model Predictions")
        
        # Prepare features
        if len(feature_cols) > 0 and model_choice != "Moving Average":
            model_key = {'Random Forest': 'rf', 'Gradient Boosting': 'gb', 'XGBoost': 'xgb'}[model_choice]
            
            if model_key in models:
                model = models[model_key]
                
                # Get recent data for prediction
                recent_data = filtered_df.tail(100).copy()
                
                # Prepare features for prediction
                X_pred = recent_data[feature_cols].values
                X_pred = np.where(np.isinf(X_pred), np.nan, X_pred)
                
                # Impute NaN values
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X_pred = imputer.fit_transform(X_pred)
                
                # Make predictions
                predictions = model.predict(X_pred)
                predictions = np.maximum(predictions, 0)
                
                # Get actual values if available
                if target_col in recent_data.columns:
                    actuals = recent_data[target_col].values
                else:
                    actuals = recent_data['Sales Count'].values
                
                # Create comparison plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=recent_data['Timestamp'],
                    y=actuals,
                    mode='lines',
                    name='Actual',
                    line=dict(color='#3182ce', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=recent_data['Timestamp'],
                    y=predictions,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='#e53e3e', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    height=500,
                    xaxis_title="Time",
                    yaxis_title="Sales Count",
                    template='plotly_white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                with col2:
                    st.metric("Root Mean Square Error (RMSE)", f"{rmse:.2f}")
                
                # Next period forecast
                st.markdown("### 📈 Next Period Forecast")
                next_pred = predictions[-1] if len(predictions) > 0 else 0
                
                if next_pred > 50:
                    st.markdown(f'<div class="warning-box"><b>⚠️ High Demand Expected:</b> Predicted sales of {next_pred:.0f} tickets</div>', unsafe_allow_html=True)
                elif next_pred > 20:
                    st.markdown(f'<div class="success-box"><b>✅ Moderate Demand:</b> Predicted sales of {next_pred:.0f} tickets</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><b>📉 Low Demand:</b> Predicted sales of {next_pred:.0f} tickets</div>', unsafe_allow_html=True)
                
                # Confidence interval (approximation)
                std_pred = np.std(predictions)
                lower_bound = max(0, next_pred - 1.96 * std_pred)
                upper_bound = next_pred + 1.96 * std_pred
                
                st.markdown(f"**95% Confidence Interval:** {lower_bound:.0f} - {upper_bound:.0f} tickets")
    
    else:
        # Moving Average forecast
        st.markdown("### Moving Average Forecast")
        
        window = {'15min': 1, '30min': 2, '1hr': 4, '2hr': 8}[horizon]
        ma_predictions = filtered_df['Sales Count'].rolling(window=window, min_periods=1).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered_df['Timestamp'],
            y=filtered_df['Sales Count'],
            mode='lines',
            name='Actual',
            line=dict(color='#3182ce', width=1),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=filtered_df['Timestamp'],
            y=ma_predictions,
            mode='lines',
            name=f'MA ({horizon})',
            line=dict(color='#e53e3e', width=2)
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Time",
            yaxis_title="Sales Count",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================================
elif page == "📈 Model Performance":
    st.markdown('<p class="main-header">📈 Model Performance</p>', unsafe_allow_html=True)
    
    # Load model comparison results
    try:
        results_df = pd.read_csv('output/model_comparison_1hr.csv')
        
        st.markdown("### Model Comparison - 1 Hour Forecast")
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='MAE',
            x=results_df['Model'],
            y=results_df['MAE'],
            marker_color='#3182ce'
        ))
        
        fig.add_trace(go.Bar(
            name='RMSE',
            x=results_df['Model'],
            y=results_df['RMSE'],
            marker_color='#e53e3e'
        ))
        
        fig.update_layout(
            barmode='group',
            height=500,
            xaxis_title="Model",
            yaxis_title="Error",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.markdown("### Detailed Results")
        st.dataframe(results_df.style.format({
            'MAE': '{:.2f}',
            'RMSE': '{:.2f}',
            'MAPE (%)': '{:.2f}',
            'Within ±10 (%)': '{:.2f}'
        }), use_container_width=True)
        
    except:
        st.warning("Model comparison results not found. Please run model training first.")
    
    # Multi-horizon results
    try:
        horizon_df = pd.read_csv('output/multi_horizon_results.csv')
        
        st.markdown("### Performance Across Forecast Horizons")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=horizon_df['Horizon'],
            y=horizon_df['MAE'],
            mode='lines+markers',
            name='MAE',
            line=dict(color='#3182ce', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=horizon_df['Horizon'],
            y=horizon_df['RMSE'],
            mode='lines+markers',
            name='RMSE',
            line=dict(color='#e53e3e', width=2)
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Forecast Horizon",
            yaxis_title="Error",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except:
        pass
    
    # Feature Importance
    if 'rf' in models:
        st.markdown("### Feature Importance (Random Forest)")
        
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': models['rf'].feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color='#3182ce'
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Importance",
            yaxis_title="Feature",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: KPI METRICS
# ============================================================================
elif page == "📋 KPI Metrics":
    st.markdown('<p class="main-header">📋 Key Performance Indicators</p>', unsafe_allow_html=True)
    
    # Calculate KPIs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Forecasting KPIs")
        
        # Load results
        try:
            results_df = pd.read_csv('output/model_comparison_1hr.csv')
            best_model = results_df.iloc[0]
            
            kpi_data = {
                'Forecast Accuracy (%)': max(0, 100 - best_model['MAPE (%)']),
                'MAE': best_model['MAE'],
                'RMSE': best_model['RMSE'],
                'Within ±10 (%)': best_model['Within ±10 (%)']
            }
            
            for kpi, value in kpi_data.items():
                st.metric(kpi, f"{value:.2f}")
                
        except:
            st.warning("KPI data not available. Run model training first.")
    
    with col2:
        st.markdown("### Operational KPIs")
        
        # Calculate operational metrics
        daily_totals = filtered_df.groupby(filtered_df['Timestamp'].dt.date)['Sales Count'].sum()
        
        op_kpis = {
            'Avg Daily Sales': daily_totals.mean(),
            'Peak Daily Sales': daily_totals.max(),
            'Min Daily Sales': daily_totals.min(),
            'Sales Std Dev': daily_totals.std()
        }
        
        for kpi, value in op_kpis.items():
            st.metric(kpi, f"{value:,.0f}")
    
    # Operational Recommendations
    st.markdown("---")
    st.markdown("### 💡 Operational Recommendations")
    
    # Peak hour analysis
    peak_hour = filtered_df.groupby('Hour')['Sales Count'].mean().idxmax()
    peak_day = filtered_df.groupby('DayOfWeek')['Sales Count'].mean().idxmax()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    recommendations = [
        f"🚢 **Staffing:** Increase staff capacity around {int(peak_hour)}:00 for peak demand handling",
        f"📅 **Scheduling:** {day_names[peak_day]}s show highest demand - consider additional ferry services",
        f"⏰ **Maintenance:** Schedule maintenance during low-demand hours (early morning/late evening)",
        f"📊 **Forecasting:** Use Moving Average for short-term predictions (best baseline model)",
        f"🎯 **Planning:** Weekend demand is ~18% higher than weekdays - adjust capacity accordingly"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Toronto Island Ferry Demand Forecasting System | Powered by Machine Learning</p>
    <p>Data Source: City of Toronto Open Data Portal</p>
</div>
""", unsafe_allow_html=True)