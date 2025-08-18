#!/usr/bin/env python3
"""
IntelliVest Methodology Page

This page explains the technical details, data sources, and methodology
behind the IntelliVest stock analysis system.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="IntelliVest Methodology",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
# 🔬 IntelliVest Methodology & Technical Details

This page explains the science behind IntelliVest - how we analyze stocks, calculate scores, and generate predictions.
""")

# Create tabs for different methodology sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Sources", 
    "📈 Technical Analysis", 
    "🤖 ML Forecasting", 
    "📊 Scoring Algorithm", 
    "🔧 System Architecture"
])

with tab1:
    st.header("📊 Data Sources & Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Real-Time Data Pipeline")
        st.markdown("""
        **Data Sources:**
        - **Yahoo Finance API**: Historical price data (OHLCV)
        - **Wikipedia**: S&P 500 ticker symbols
        - **Supabase PostgreSQL**: Cloud database storage
        
        **Data Collection:**
        - **Automated ETL**: Daily updates via GitHub Actions
        - **5+ Years History**: Sufficient data for technical analysis
        - **Real-time Updates**: Market close data available next day
        """)
        
        st.info("💡 **Data Quality**: We use Yahoo Finance's reliable API which provides clean, adjusted stock data with proper handling of splits and dividends.")
    
    with col2:
        st.subheader("🗄️ Database Architecture")
        st.markdown("""
        **Storage:**
        - **Cloud Database**: Supabase PostgreSQL
        - **Connection Pooling**: Reliable cloud access
        - **Indexed Tables**: Fast query performance
        
        **Schema:**
        ```sql
        daily_prices (
            ticker VARCHAR,
            date DATE,
            open DECIMAL,
            high DECIMAL,
            low DECIMAL,
            close DECIMAL,
            volume BIGINT,
            PRIMARY KEY (ticker, date)
        )
        ```
        """)

with tab2:
    st.header("📈 Technical Analysis Engine")
    
    st.subheader("🔍 Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 RSI (Relative Strength Index)**
        - **Period**: 14 days
        - **Purpose**: Identify overbought/oversold conditions
        - **Interpretation**:
          - RSI < 30: Oversold (bullish signal)
          - RSI > 70: Overbought (bearish signal)
          - 30-70: Neutral range
        
        **📈 Moving Averages**
        - **SMA 50**: 50-day simple moving average
        - **SMA 200**: 200-day simple moving average
        - **Golden Cross**: SMA50 > SMA200 (bullish)
        - **Death Cross**: SMA50 < SMA200 (bearish)
        """)
    
    with col2:
        st.markdown("""
        **🔄 MACD (Moving Average Convergence Divergence)**
        - **Parameters**: 12, 26, 9 (fast, slow, signal)
        - **Components**:
          - MACD Line: Fast MA - Slow MA
          - Signal Line: 9-day EMA of MACD Line
          - Histogram: MACD Line - Signal Line
        - **Signals**:
          - MACD > Signal: Bullish momentum
          - MACD < Signal: Bearish momentum
        
        **📊 Volume Analysis**
        - **Volume Confirmation**: High volume validates price moves
        - **Volume Divergence**: Price up, volume down = weak move
        """)
    
    # Create a sample technical indicators visualization
    st.subheader("📊 Sample Technical Indicators")
    
    # Generate sample data for demonstration
    import numpy as np
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    # Calculate sample indicators
    rsi = 50 + 20 * np.sin(np.arange(100) * 0.1)
    sma_50 = prices.rolling(50).mean()
    sma_200 = prices.rolling(200).mean()
    
    # Create chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Price & Moving Averages', 'RSI', 'Sample Data'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=dates, y=prices, name='Price', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sma_50, name='SMA 50', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sma_200, name='SMA 200', line=dict(color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=dates, y=rsi, name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.add_trace(
        go.Scatter(x=dates, y=np.random.randn(100), name='Sample Data', line=dict(color='green')),
        row=3, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🤖 Machine Learning Forecasting")
    
    st.markdown("""
    ## 🔮 Facebook Prophet Model
    
    **Why Prophet?**
    - **Robust**: Handles missing data and outliers gracefully
    - **Automatic**: Detects trends, seasonality, and changepoints
    - **Uncertainty**: Provides confidence intervals for predictions
    - **Interpretable**: Easy to understand and explain
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Model Architecture")
        st.markdown("""
        **Core Components:**
        - **Trend**: Flexible trend modeling with changepoints
        - **Seasonality**: Weekly, monthly, quarterly patterns
        - **Holidays**: Market-specific event effects
        - **Noise**: Random fluctuations and uncertainty
        
        **Parameters:**
        - **Changepoint Prior Scale**: Trend flexibility (0.05)
        - **Seasonality Prior Scale**: Seasonality strength (10.0)
        - **Holidays Prior Scale**: Holiday effect strength (10.0)
        """)
    
    with col2:
        st.subheader("📊 Forecasting Process")
        st.markdown("""
        **1. Data Preparation**
        - Clean historical data
        - Handle missing values
        - Format for Prophet (ds, y columns)
        
        **2. Model Training**
        - Fit on historical data
        - Detect patterns automatically
        - Optimize parameters
        
        **3. Prediction Generation**
        - Extend timeline
        - Generate forecasts
        - Calculate confidence intervals
        """)
    
    st.subheader("🎯 Model Validation")
    st.markdown("""
    **Accuracy Metrics:**
    - **MAPE**: Mean Absolute Percentage Error
    - **RMSE**: Root Mean Square Error
    - **Confidence Intervals**: 90%, 95%, 99% levels
        
    **Important Notes:**
    - ⚠️ **Past performance doesn't guarantee future results**
    - 📈 **Forecasts are probabilistic, not deterministic**
    - 🔄 **Models should be retrained regularly**
    - 📊 **Use as one of many analysis tools**
    """)

with tab4:
    st.header("📊 Scoring Algorithm")
    
    st.markdown("""
    ## 🎯 Technical Score Calculation (0-100)
    
    Our scoring system combines multiple technical indicators to provide a comprehensive
    assessment of a stock's current technical position.
    """)
    
    # Create scoring breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 RSI Analysis (30 points max)")
        
        # Create RSI scoring visualization
        rsi_values = np.linspace(0, 100, 100)
        rsi_scores = np.where(rsi_values < 30, 30, 
                             np.where(rsi_values > 70, -10, 0))
        
        fig = px.line(x=rsi_values, y=rsi_scores, 
                     title="RSI Scoring Function",
                     labels={'x': 'RSI Value', 'y': 'Score Points'})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        - **RSI < 30**: +30 points (oversold - bullish)
        - **RSI > 70**: -10 points (overbought - bearish)
        - **30 ≤ RSI ≤ 70**: 0 points (neutral)
        """)
    
    with col2:
        st.subheader("📊 Moving Average Analysis (30 points max)")
        
        # Create SMA scoring visualization
        sma_diff = np.linspace(-20, 20, 100)
        sma_scores = np.where(sma_diff > 0, 30, 0)
        
        fig = px.line(x=sma_diff, y=sma_scores,
                     title="SMA Crossover Scoring",
                     labels={'x': 'SMA50 - SMA200 (%)', 'y': 'Score Points'})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="blue", annotation_text="Golden Cross")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        - **SMA50 > SMA200**: +30 points (golden cross)
        - **SMA50 ≤ SMA200**: 0 points (death cross)
        """)
    
    st.subheader("🔄 MACD Momentum (20 points max)")
    st.markdown("""
    - **MACD Line > Signal Line**: +20 points (bullish momentum)
    - **MACD Line ≤ Signal Line**: 0 points (bearish momentum)
    """)
    
    st.subheader("📈 Price vs Long-term MA (20 points max)")
    st.markdown("""
    - **Close > SMA200**: +20 points (above long-term trend)
    - **Close ≤ SMA200**: 0 points (below long-term trend)
    """)
    
    # Final scoring breakdown
    st.subheader("🎯 Final Score Interpretation")
    
    score_ranges = {
        "85-100": "Strong Buy 🚀",
        "70-84": "Buy 📈", 
        "30-69": "Hold ⏸️",
        "0-29": "Consider Selling 📉"
    }
    
    for range_str, recommendation in score_ranges.items():
        st.markdown(f"- **{range_str}**: {recommendation}")

with tab5:
    st.header("🔧 System Architecture")
    
    st.markdown("""
    ## 🏗️ Technical Infrastructure
    
    IntelliVest is built with modern, scalable technologies following industry best practices.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖥️ Backend Technologies")
        st.markdown("""
        **Core Framework:**
        - **Python 3.11+**: Modern Python with type hints
        - **Streamlit**: Rapid web app development
        - **Plotly**: Interactive data visualization
        
        **Data Processing:**
        - **Pandas**: Data manipulation and analysis
        - **NumPy**: Numerical computing
        - **TA-Lib**: Technical analysis indicators
        
        **Machine Learning:**
        - **Prophet**: Time series forecasting
        - **Scikit-learn**: Additional ML capabilities
        """)
    
    with col2:
        st.subheader("🗄️ Data Infrastructure")
        st.markdown("""
        **Database:**
        - **PostgreSQL**: Robust relational database
        - **Supabase**: Cloud-hosted with connection pooling
        - **Connection Pooling**: Reliable cloud access
        
        **ETL Pipeline:**
        - **GitHub Actions**: Automated daily updates
        - **Yahoo Finance API**: Real-time data collection
        - **Error Handling**: Graceful failure recovery
        """)
    
    st.subheader("🚀 Deployment & Scaling")
    st.markdown("""
    **Cloud Platform:**
    - **Streamlit Cloud**: Hosted web application
    - **Supabase**: Cloud database and authentication
    - **GitHub**: Version control and CI/CD
    
    **Scalability Features:**
    - **Connection Pooling**: Handle multiple users
    - **Caching**: Optimize repeated calculations
    - **Async Processing**: Non-blocking operations
    """)
    
    st.subheader("🔒 Security & Best Practices")
    st.markdown("""
    **Security Measures:**
    - **Environment Variables**: No hardcoded credentials
    - **Parameterized Queries**: SQL injection prevention
    - **Input Validation**: User input sanitization
    
    **Code Quality:**
    - **Type Hints**: Better code documentation
    - **Error Handling**: Comprehensive exception management
    - **Logging**: Detailed operation tracking
    - **Testing**: Unit and integration tests
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 IntelliVest Methodology | Built with ❤️ using Streamlit & Python</p>
    <p><small>This information is for educational purposes only. Always conduct your own research.</small></p>
</div>
""", unsafe_allow_html=True)
