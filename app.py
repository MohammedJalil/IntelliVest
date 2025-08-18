#!/usr/bin/env python3
"""
IntelliVest Stock Analyzer - Streamlit Web Application

This Streamlit app provides a user interface for the IntelliVest stock analysis project,
allowing users to analyze stocks, view technical indicators, and get trading recommendations.
"""

import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta

# Import IntelliVest functions
from analysis_engine import (
    calculate_technical_indicators, 
    get_technical_score, 
    get_final_recommendation
)

# Import ML forecasting
try:
    from ml_forecaster import forecast_stock_price, StockForecaster
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ ML forecasting not available. Install Prophet: `pip install prophet`")

# Page configuration
st.set_page_config(
    page_title="IntelliVest Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add navigation info
st.sidebar.markdown("---")
st.sidebar.markdown("**📚 Learn More:**")
st.sidebar.markdown("Check out the **Methodology** page to understand how our analysis works!")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-buy {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .recommendation-sell {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    .recommendation-hold {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

def get_database_connection():
    """
    Create and return a connection to the PostgreSQL database.
    
    Returns:
        psycopg2.extensions.connection: Database connection object
    """
    try:
        # Try connection string first, fall back to individual parameters
        if "DATABASE_URL" in st.secrets:
            connection = psycopg2.connect(st.secrets["DATABASE_URL"])
        else:
            connection = psycopg2.connect(
                host=st.secrets["DB_HOST"],
                database=st.secrets["DB_NAME"],
                user=st.secrets["DB_USER"],
                password=st.secrets["DB_PASS"],
                port=st.secrets["DB_PORT"]
            )
        return connection
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def fetch_stock_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch historical stock data from the PostgreSQL database.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days of data to fetch
    
    Returns:
        pd.DataFrame: DataFrame with stock data or empty DataFrame if error
    """
    try:
        connection = get_database_connection()
        if connection is None:
            return pd.DataFrame()
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        query = """
        SELECT ticker, date, open, high, low, close, volume
        FROM daily_prices 
        WHERE ticker = %s AND date >= %s AND date <= %s
        ORDER BY date ASC
        """
        
        df = pd.read_sql_query(
            query, 
            connection, 
            params=(ticker.upper(), start_date, end_date)
        )
        
        connection.close()
        
        if df.empty:
            st.warning(f"No data found for ticker {ticker.upper()}")
            return pd.DataFrame()
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def create_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create an interactive candlestick chart with moving averages.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        ticker (str): Stock ticker symbol
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Calculate moving averages
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{ticker.upper()} Price Chart', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['SMA_200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker.upper()} Stock Analysis',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def create_technical_indicators_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create a chart showing technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        ticker (str): Stock ticker symbol
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Create subplots for different indicators
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('RSI', 'MACD', 'Price vs Moving Averages'),
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['RSI_14'],
                mode='lines',
                name='RSI 14',
                line=dict(color='purple', width=2)
            ),
            row=1, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['MACD_12_26_9'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['MACDs_12_26_9'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Price vs Moving Averages
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['SMA_200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker.upper()} Technical Indicators',
            height=600,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating technical indicators chart: {e}")
        return None

def display_recommendation(recommendation: str, score: int):
    """
    Display the recommendation with appropriate styling.
    
    Args:
        recommendation (str): The recommendation string
        score (int): The technical score
    """
    # Determine CSS class based on recommendation
    if "Buy" in recommendation:
        css_class = "recommendation-buy"
    elif "Sell" in recommendation:
        css_class = "recommendation-sell"
    else:
        css_class = "recommendation-hold"
    
    st.markdown(f"""
    <div class="metric-card {css_class}">
        <h3>Recommendation: {recommendation}</h3>
        <p>Technical Score: {score}/100</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 IntelliVest Stock Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Stock Analysis")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "Enter Stock Ticker:",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, GOOGL",
        help="Enter a valid stock ticker symbol"
    ).upper().strip()
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Analyze Stock", type="primary")
    
    # Additional options
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Analysis Period:**")
    days = st.sidebar.slider("Days of Data:", min_value=30, max_value=1095, value=365, step=30)
    
    # Main content area
    if analyze_button and ticker:
        if not ticker:
            st.error("Please enter a valid stock ticker.")
            return
        
        # Show loading spinner
        with st.spinner(f"Analyzing {ticker}..."):
            
            # Fetch data
            df = fetch_stock_data(ticker, days)
            
            if df.empty:
                st.error(f"Could not fetch data for {ticker}. Please check if the ticker is valid and data exists in the database.")
                return
            
            # Calculate technical indicators
            try:
                df_with_indicators = calculate_technical_indicators(df)
                
                if df_with_indicators.empty:
                    st.error("Could not calculate technical indicators. Insufficient data.")
                    return
                
                # Get latest data for scoring
                latest_data = df_with_indicators.iloc[-1]
                
                # Calculate technical score
                technical_score = get_technical_score(latest_data)
                
                # Get recommendation
                recommendation = get_final_recommendation(technical_score)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${latest_data['close']:.2f}")
                
                with col2:
                    st.metric("Technical Score", f"{technical_score}/100")
                
                with col3:
                    st.metric("Recommendation", recommendation)
                
                # Display recommendation with styling
                st.markdown("---")
                display_recommendation(recommendation, technical_score)
                
                # Create tabs for different views
                tab_names = ["📊 Price Chart", "📈 Technical Indicators", "🔮 ML Forecast", "📋 Data Table", "ℹ️ Analysis Summary"]
                tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)
                
                with tab1:
                    st.subheader("Price Chart with Moving Averages")
                    candlestick_fig = create_candlestick_chart(df_with_indicators, ticker)
                    if candlestick_fig:
                        st.plotly_chart(candlestick_fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Technical Indicators")
                    indicators_fig = create_technical_indicators_chart(df_with_indicators, ticker)
                    if indicators_fig:
                        st.plotly_chart(indicators_fig, use_container_width=True)
                
                with tab3:
                    st.subheader("🔮 Machine Learning Price Forecast")
                    
                    if ML_AVAILABLE:
                        # Forecast options
                        col1, col2 = st.columns(2)
                        with col1:
                            forecast_days = st.selectbox(
                                "Forecast Period:",
                                options=[7, 14, 30, 60],
                                index=2,
                                help="Number of days to forecast into the future"
                            )
                        
                        with col2:
                            confidence_level = st.selectbox(
                                "Confidence Level:",
                                options=[0.90, 0.95, 0.99],
                                index=1,
                                help="Confidence interval for predictions"
                            )
                        
                        # Generate forecast button
                        if st.button("🚀 Generate Forecast", type="primary"):
                            with st.spinner("Training ML model and generating forecast..."):
                                try:
                                    # Generate forecast
                                    forecaster, forecast_chart = forecast_stock_price(
                                        df_with_indicators, 
                                        ticker, 
                                        forecast_days=forecast_days,
                                        confidence_interval=confidence_level
                                    )
                                    
                                    # Display forecast chart
                                    st.plotly_chart(forecast_chart, use_container_width=True)
                                    
                                    # Display forecast summary
                                    st.subheader("📊 Forecast Summary")
                                    summary = forecaster.get_forecast_summary()
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Trend Direction", summary['trend_direction'])
                                    with col2:
                                        st.metric("Volatility", summary['volatility_estimate'])
                                    with col3:
                                        st.metric("Confidence", summary['confidence_interval'])
                                    
                                    # Show next 7 days predictions
                                    st.subheader("📅 Next 7 Days Predictions")
                                    predictions_df = pd.DataFrame({
                                        'Date': summary['next_7_days']['dates'],
                                        'Predicted Price': summary['next_7_days']['predictions'],
                                        'Lower Bound': summary['next_7_days']['lower_bound'],
                                        'Upper Bound': summary['next_7_days']['upper_bound']
                                    })
                                    st.dataframe(predictions_df, use_container_width=True)
                                    
                                    # Model insights
                                    with st.expander("🔍 Model Insights"):
                                        insights = forecaster.get_model_insights()
                                        st.write(f"**Trend Strength:** {insights.get('trend_strength', 'N/A')}")
                                        st.write(f"**Seasonality Detected:** {insights.get('seasonality_detected', 'N/A')}")
                                        st.write(f"**Changepoints:** {insights.get('changepoints', 'N/A')}")
                                        
                                        st.info("💡 **About the Model:** This forecast uses Facebook Prophet, a powerful time-series forecasting model that automatically detects trends, seasonality, and changepoints in your data.")
                                    
                                except Exception as e:
                                    st.error(f"Forecast generation failed: {e}")
                                    st.info("This might be due to insufficient data or model training issues.")
                    else:
                        st.error("ML forecasting is not available. Please install Prophet library.")
                        st.code("pip install prophet")
                
                with tab4:
                    st.subheader("Recent Data")
                    # Show last 10 rows of data
                    display_df = df_with_indicators[['date', 'close', 'RSI_14', 'SMA_50', 'SMA_200', 'MACD_12_26_9']].tail(10)
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    display_df = display_df.round(2)
                    st.dataframe(display_df, use_container_width=True)
                
                with tab5:
                    st.subheader("Analysis Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Latest Technical Indicators:**")
                        st.write(f"• **RSI (14):** {latest_data['RSI_14']:.2f}")
                        st.write(f"• **SMA (50):** ${latest_data['SMA_50']:.2f}")
                        st.write(f"• **SMA (200):** ${latest_data['SMA_200']:.2f}")
                        st.write(f"• **MACD:** {latest_data['MACD_12_26_9']:.4f}")
                        st.write(f"• **MACD Signal:** {latest_data['MACDs_12_26_9']:.4f}")
                    
                    with col2:
                        st.markdown("**Market Position:**")
                        if latest_data['close'] > latest_data['SMA_200']:
                            st.success("✅ Above 200-day MA (Bullish)")
                        else:
                            st.error("❌ Below 200-day MA (Bearish)")
                        
                        if latest_data['SMA_50'] > latest_data['SMA_200']:
                            st.success("✅ Golden Cross (SMA50 > SMA200)")
                        else:
                            st.error("❌ Death Cross (SMA50 < SMA200)")
                        
                        if latest_data['RSI_14'] < 30:
                            st.success("✅ Oversold (RSI < 30)")
                        elif latest_data['RSI_14'] > 70:
                            st.warning("⚠️ Overbought (RSI > 70)")
                        else:
                            st.info("ℹ️ Neutral RSI")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("This might be due to insufficient data for the selected time period.")
    
    # Default view when no analysis is performed
    else:
        st.markdown("""
        ## Welcome to IntelliVest Stock Analyzer! 🚀
        
        **How to use:**
        1. Enter a stock ticker symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)
        2. Click the "Analyze Stock" button
        3. View technical analysis, charts, and recommendations
        
        **Features:**
        - 📊 Interactive price charts with moving averages
        - 📈 Technical indicators (RSI, MACD, SMAs)
        - 🎯 AI-powered trading recommendations
        - 📋 Detailed analysis summary
        
        **Supported Indicators:**
        - RSI (Relative Strength Index)
        - SMA 50 & 200 (Simple Moving Averages)
        - MACD (Moving Average Convergence Divergence)
        - Technical scoring system (0-100)
        
        **Data Source:** Historical stock data from your IntelliVest database
        """)
        
        # Show some example tickers
        st.markdown("**Popular Tickers to Try:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("AAPL"):
                st.session_state.ticker = "AAPL"
                st.rerun()
        
        with col2:
            if st.button("MSFT"):
                st.session_state.ticker = "MSFT"
                st.rerun()
        
        with col3:
            if st.button("GOOGL"):
                st.session_state.ticker = "GOOGL"
                st.rerun()
        
        with col4:
            if st.button("TSLA"):
                st.session_state.ticker = "TSLA"
                st.rerun()

if __name__ == "__main__":
    main()
