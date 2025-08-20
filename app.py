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

from datetime import datetime, timedelta

# Import IntelliVest functions
from analysis_engine import (
    calculate_technical_indicators, 
    get_technical_score, 
    get_final_recommendation
)

# Import ML forecasting
try:
    from ml_forecaster import forecast_stock_price
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
        # Try Streamlit secrets first (for Streamlit Cloud deployment)
        if hasattr(st, 'secrets') and st.secrets:
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
        else:
            # Fall back to environment variables (for local development)
            import os
            
            if os.getenv('DATABASE_URL'):
                connection = psycopg2.connect(os.getenv('DATABASE_URL'))
            else:
                # Check if all required environment variables are present
                required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASS', 'DB_PORT']
                missing_vars = [var for var in required_vars if not os.getenv(var)]
                
                if missing_vars:
                    st.error(f"Missing required environment variables: {missing_vars}")
                    st.info("Please create a .env file with your database credentials or set the environment variables.")
                    return None
                
                connection = psycopg2.connect(
                    host=os.getenv('DB_HOST'),
                    database=os.getenv('DB_NAME'),
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASS'),
                    port=int(os.getenv('DB_PORT', '5432'))
                )
        
        return connection
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.info("Please check your database credentials and ensure the database is running.")
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
            params=[ticker.upper(), start_date, end_date]
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

def create_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure | None:
    """
    Create an interactive candlestick chart with moving averages.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        ticker (str): Stock ticker symbol
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Validate and convert data types to ensure numeric values
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check if we have enough data after cleaning
        if len(df) < 20:
            st.warning(f"Insufficient data for {ticker} candlestick chart (need at least 20 data points)")
            return None
        
        # Calculate moving averages (this will handle NaN values gracefully)
        df['SMA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['SMA_200'] = df['close'].rolling(window=200, min_periods=1).mean()
        
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

def create_technical_indicators_chart(df: pd.DataFrame, ticker: str) -> go.Figure | None:
    """
    Create a chart showing technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        ticker (str): Stock ticker symbol
    
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        # Validate and convert data types to ensure numeric values
        required_columns = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'close', 'SMA_50', 'SMA_200']
        
        for col in required_columns:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check if we have enough data after cleaning
        if len(df) < 5:
            st.warning(f"Insufficient data for {ticker} technical indicators chart (need at least 5 data points)")
            return None
        
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
        fig.add_hline(y=70, line_dash="dash", line_color="red", row="1", col="1")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row="1", col="1")
        
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
    """Display recommendation with appropriate styling."""
    if recommendation == "Buy":
        css_class = "recommendation-buy"
    elif recommendation == "Sell":
        css_class = "recommendation-sell"
    else:
        css_class = "recommendation-hold"
    
    st.markdown(f"""
    <div class="metric-card {css_class}">
        <h3>Recommendation: {recommendation}</h3>
        <p>Technical Score: {score}/100</p>
    </div>
    """, unsafe_allow_html=True)

def run_stock_screener(stock_list: list, days: int, min_threshold: int) -> list:
    """
    Run stock screener to analyze multiple stocks and return top performers.
    
    Args:
        stock_list (list): List of stock tickers to analyze
        days (int): Number of days to analyze for each stock
        min_threshold (int): Minimum technical score threshold
        
    Returns:
        list: List of stock analysis results above threshold
    """
    results = []
    
    # Progress bar for user feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(stock_list):
        try:
            # Update progress
            progress = (i + 1) / len(stock_list)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {ticker}... ({i+1}/{len(stock_list)})")
            
            # Fetch data for this stock
            df = fetch_stock_data(ticker, days)
            
            if df.empty:
                continue  # Skip stocks with no data
            
            # Calculate technical indicators
            try:
                df_with_indicators = calculate_technical_indicators(df)
                
                if df_with_indicators.empty:
                    continue  # Skip stocks with insufficient data
                
                # Get latest data for scoring
                latest_data = df_with_indicators.iloc[-1]
                
                # Calculate technical score
                technical_score = get_technical_score(latest_data)
                
                # Only include stocks above threshold
                if technical_score >= min_threshold:
                    # Get recommendation
                    recommendation = get_final_recommendation(technical_score)
                    
                    # Determine trend (above/below 200-day MA)
                    above_200ma = latest_data['close'] > latest_data['SMA_200']
                    
                    # Add to results
                    results.append({
                        'ticker': ticker,
                        'technical_score': technical_score,
                        'recommendation': recommendation,
                        'current_price': latest_data['close'],
                        'rsi': latest_data['RSI_14'],
                        'above_200ma': above_200ma,
                        'sma_50': latest_data['SMA_50'],
                        'sma_200': latest_data['SMA_200'],
                        'macd': latest_data['MACD_12_26_9']
                    })
                    
            except Exception as e:
                # Log error but continue with other stocks
                st.warning(f"Could not analyze {ticker}: {e}")
                continue
                
        except Exception as e:
            # Log error but continue with other stocks
            st.warning(f"Could not fetch data for {ticker}: {e}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Sort results by technical score (highest first)
    results.sort(key=lambda x: x['technical_score'], reverse=True)
    
    return results

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 IntelliVest Stock Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🚀 Stock Analysis")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "Enter Stock Ticker:",
        value=st.session_state.get('current_ticker', 'AAPL'),
        placeholder="e.g., AAPL, MSFT, GOOGL",
        help="Enter a valid stock ticker symbol",
        key="ticker_input"
    ).upper().strip()
    
    # Analysis button
    analyze_button = st.sidebar.button("🔍 Analyze Stock", type="primary")
    
    # Helpful note about Enter key
    st.sidebar.info("💡 **Quick Search:** Type ticker → Press Enter → Click 'Analyze Stock'")
    
    # Handle Enter key submission by monitoring ticker changes
    if ticker and ticker != st.session_state.get('current_ticker', ''):
        st.session_state.current_ticker = ticker
        st.session_state.should_analyze = True
    
    # Check if we should analyze (from popular ticker buttons or Enter key)
    should_analyze = analyze_button or st.session_state.get('should_analyze', False)
    
    # Reset the flag after using it
    if st.session_state.get('should_analyze', False):
        st.session_state.should_analyze = False
    
    # Additional options
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📅 Analysis Period:**")
    days = st.sidebar.slider("Days of Data:", min_value=30, max_value=1095, value=365, step=30)
    
    # Store in session state to prevent page refresh issues
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = ticker
    if 'current_days' not in st.session_state:
        st.session_state.current_days = days
    
    # Main content area - Show when we have data or are analyzing
    has_data = (st.session_state.get('current_df') is not None and 
                st.session_state.get('current_df_with_indicators') is not None and
                st.session_state.get('current_ticker') is not None)
    
    if should_analyze and ticker:
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
            
            # Store data in session state for persistence
            st.session_state.current_df = df
            st.session_state.current_ticker = ticker
            st.session_state.current_days = days
            
            # Calculate technical indicators
            try:
                df_with_indicators = calculate_technical_indicators(df)
                
                if df_with_indicators.empty:
                    st.error("Could not calculate technical indicators. Insufficient data.")
                    return
                
                # Store indicators data in session state
                st.session_state.current_df_with_indicators = df_with_indicators
                
                # Get latest data for scoring
                latest_data = df_with_indicators.iloc[-1]
                
                # Calculate technical score
                technical_score = get_technical_score(latest_data)
                
                # Get recommendation
                recommendation = get_final_recommendation(technical_score)
                
                # Store results in session state
                st.session_state.current_technical_score = technical_score
                st.session_state.current_recommendation = recommendation
                st.session_state.current_latest_data = latest_data
                
                st.success(f"✅ Analysis completed for {ticker}")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("This might be due to insufficient data for the selected time period.")
                return
    
    # Display results if we have data (either from current analysis or stored in session state)
    if has_data:
        # Get data from session state
        current_df = st.session_state.get('current_df')
        current_df_with_indicators = st.session_state.get('current_df_with_indicators')
        current_ticker = st.session_state.get('current_ticker')
        current_technical_score = st.session_state.get('current_technical_score')
        current_recommendation = st.session_state.get('current_recommendation')
        current_latest_data = st.session_state.get('current_latest_data')
        
        # Validate that we have all required data
        if (current_df is not None and current_df_with_indicators is not None and 
            current_ticker is not None and current_technical_score is not None and
            current_recommendation is not None and current_latest_data is not None):
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_latest_data['close']:.2f}")
            
            with col2:
                st.metric("Technical Score", f"{current_technical_score}/100")
            
            with col3:
                st.metric("Recommendation", current_recommendation)
            
            # Display recommendation with styling
            st.markdown("---")
            display_recommendation(current_recommendation, current_technical_score)
            
            # Create tabs for different views
            tab_names = ["📊 Price Chart", "📈 Technical Indicators", "🔮 ML Forecast", "📋 Data Table", "ℹ️ Analysis Summary", "🔍 Stock Screener"]
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)
            
            with tab1:
                st.subheader("Price Chart with Moving Averages")
                candlestick_fig = create_candlestick_chart(current_df_with_indicators, current_ticker)
                if candlestick_fig:
                    st.plotly_chart(candlestick_fig, use_container_width=True)
            
            with tab2:
                st.subheader("Technical Indicators")
                indicators_fig = create_technical_indicators_chart(current_df_with_indicators, current_ticker)
                if indicators_fig:
                    st.plotly_chart(indicators_fig, use_container_width=True)
            
            with tab3:
                st.subheader("🔮 Machine Learning Price Forecast")
                
                if ML_AVAILABLE:
                    # Store forecast state to prevent page refresh
                    if 'forecast_generated' not in st.session_state:
                        st.session_state.forecast_generated = False
                    if 'forecast_data' not in st.session_state:
                        st.session_state.forecast_data = None
                    
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
                    if st.button("🚀 Generate Forecast", type="primary", key="generate_forecast"):
                        st.session_state.forecast_generated = True
                        with st.spinner("Training ML model and generating forecast..."):
                            try:
                                # Generate forecast using session state data
                                forecaster, forecast_chart = forecast_stock_price(
                                    current_df_with_indicators, 
                                    current_ticker, 
                                    forecast_days=forecast_days,
                                    confidence_interval=confidence_level
                                )
                                
                                # Store forecast data in session state
                                st.session_state.forecast_data = {
                                    'forecaster': forecaster,
                                    'forecast_chart': forecast_chart,
                                    'ticker': current_ticker
                                }
                                
                                st.success("✅ Forecast generated successfully!")
                                
                            except Exception as e:
                                st.error(f"Forecast generation failed: {e}")
                                st.info("This might be due to insufficient data or model training issues.")
                                st.session_state.forecast_generated = False
                    
                    # Display forecast results if available
                    if st.session_state.forecast_generated and st.session_state.forecast_data:
                        forecast_data = st.session_state.forecast_data
                        
                        # Display forecast chart
                        st.plotly_chart(forecast_data['forecast_chart'], use_container_width=True)
                        
                        # Display forecast summary
                        st.subheader("📊 Forecast Summary")
                        summary = forecast_data['forecaster'].get_forecast_summary()
                        
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
                            insights = forecast_data['forecaster'].get_model_insights()
                            st.write(f"**Trend Strength:** {insights.get('trend_strength', 'N/A')}")
                            st.write(f"**Seasonality Detected:** {insights.get('seasonality_detected', 'N/A')}")
                            st.write(f"**Changepoints:** {insights.get('changepoints', 'N/A')}")
                            
                            st.info("💡 **About the Model:** This forecast uses Facebook Prophet, a powerful time-series forecasting model that automatically detects trends, seasonality, and changepoints in your data.")
                        
                        # Clear forecast button
                        if st.button("🗑️ Clear Forecast", key="clear_forecast"):
                            st.session_state.forecast_generated = False
                            st.session_state.forecast_data = None
                            st.rerun()
                else:
                    st.error("ML forecasting is not available. Please install Prophet library.")
                    st.code("pip install prophet")
            
            with tab4:
                st.subheader("Recent Data")
                # Show last 10 rows of data
                display_df = current_df_with_indicators[['date', 'close', 'RSI_14', 'SMA_50', 'SMA_200', 'MACD_12_26_9']].tail(10)
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                display_df = display_df.round(2)
                st.dataframe(display_df, use_container_width=True)
            
            with tab5:
                st.subheader("Analysis Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Latest Technical Indicators:**")
                    st.write(f"• **RSI (14):** {current_latest_data['RSI_14']:.2f}")
                    st.write(f"• **SMA (50):** ${current_latest_data['SMA_50']:.2f}")
                    st.write(f"• **SMA (200):** ${current_latest_data['SMA_200']:.2f}")
                    st.write(f"• **MACD:** {current_latest_data['MACD_12_26_9']:.4f}")
                    st.write(f"• **MACD Signal:** {current_latest_data['MACDs_12_26_9']:.4f}")
                
                with col2:
                    st.markdown("**Market Position:**")
                    if current_latest_data['close'] > current_latest_data['SMA_200']:
                        st.success("✅ Above 200-day MA (Bullish)")
                    else:
                        st.error("❌ Below 200-day MA (Bearish)")
                    
                    if current_latest_data['SMA_50'] > current_latest_data['SMA_200']:
                        st.success("✅ Golden Cross (SMA50 > SMA200)")
                    else:
                        st.error("❌ Death Cross (SMA50 < SMA200)")
                    
                    if current_latest_data['RSI_14'] < 30:
                        st.success("✅ Oversold (RSI < 30)")
                    elif current_latest_data['RSI_14'] > 70:
                        st.warning("⚠️ Overbought (RSI > 70)")
                    else:
                        st.info("ℹ️ Neutral RSI")
            
            with tab6:
                st.subheader("🔍 Stock Screener")
                st.markdown("**Automatically discover top-performing stocks based on technical analysis scores.**")
                
                # Stock screener options
                col1, col2 = st.columns(2)
                with col1:
                    screener_days = st.selectbox(
                        "Analysis Period:",
                        options=[30, 60, 90, 180, 365],
                        index=4,
                        help="Number of days to analyze for each stock"
                    )
                
                with col2:
                    min_score_threshold = st.slider(
                        "Minimum Score Threshold:",
                        min_value=0,
                        max_value=100,
                        value=50,
                        help="Only show stocks with technical scores above this threshold"
                    )
                
                # Predefined list of popular stocks to screen
                popular_stocks = [
                    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC",
                    "JPM", "JNJ", "PG", "UNH", "HD", "DIS", "PYPL", "CRM", "ADBE", "ORCL",
                    "IBM", "CSCO", "TXN", "QCOM", "AVGO", "TMO", "ABT", "LLY", "PFE", "MRK",
                    "KO", "PEP", "WMT", "COST", "TGT", "LOW", "SBUX", "NKE", "MCD", "BA"
                ]
                
                # Stock screener button
                if st.button("🚀 Run Stock Screener", type="primary", key="run_screener"):
                    with st.spinner("Analyzing multiple stocks... This may take a few minutes."):
                        try:
                            # Run the stock screener
                            screener_results = run_stock_screener(popular_stocks, screener_days, min_score_threshold)
                            
                            if screener_results:
                                # Store results in session state
                                st.session_state.screener_results = screener_results
                                st.success(f"✅ Screener completed! Found {len(screener_results)} stocks above threshold.")
                            else:
                                st.warning("⚠️ No stocks found above the minimum score threshold.")
                                
                        except Exception as e:
                            st.error(f"Screener failed: {e}")
                            st.info("This might be due to database connection issues or insufficient data.")
                
                # Display screener results if available
                if 'screener_results' in st.session_state and st.session_state.screener_results:
                    screener_results = st.session_state.screener_results
                    
                    # Group stocks by recommendation category
                    buy_stocks = [s for s in screener_results if s['recommendation'] == 'Buy']
                    hold_stocks = [s for s in screener_results if s['recommendation'] == 'Hold']
                    sell_stocks = [s for s in screener_results if s['recommendation'] == 'Sell']
                    
                    # Display top performers in each category
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 🟢 **Top Buy Recommendations**")
                        if buy_stocks:
                            # Sort by technical score (highest first)
                            top_buy = sorted(buy_stocks, key=lambda x: x['technical_score'], reverse=True)[:5]
                            for i, stock in enumerate(top_buy, 1):
                                with st.container():
                                    st.markdown(f"""
                                    **{i}. {stock['ticker']}** - Score: **{stock['technical_score']}/100**
                                    - Price: ${stock['current_price']:.2f}
                                    - RSI: {stock['rsi']:.1f}
                                    - Trend: {'🟢 Bullish' if stock['above_200ma'] else '🔴 Bearish'}
                                    """)
                                    if st.button(f"Analyze {stock['ticker']}", key=f"analyze_{stock['ticker']}"):
                                        st.session_state.current_ticker = stock['ticker']
                                        st.session_state.should_analyze = True
                                        st.rerun()
                        else:
                            st.info("No Buy recommendations found.")
                    
                    with col2:
                        st.markdown("### 🟡 **Top Hold Recommendations**")
                        if hold_stocks:
                            # Sort by technical score (highest first)
                            top_hold = sorted(hold_stocks, key=lambda x: x['technical_score'], reverse=True)[:5]
                            for i, stock in enumerate(top_hold, 1):
                                with st.container():
                                    st.markdown(f"""
                                    **{i}. {stock['ticker']}** - Score: **{stock['technical_score']}/100**
                                    - Price: ${stock['current_price']:.2f}
                                    - RSI: {stock['rsi']:.1f}
                                    - Trend: {'🟢 Bullish' if stock['above_200ma'] else '🔴 Bearish'}
                                    """)
                                    if st.button(f"Analyze {stock['ticker']}", key=f"analyze_hold_{stock['ticker']}"):
                                        st.session_state.current_ticker = stock['ticker']
                                        st.session_state.should_analyze = True
                                        st.rerun()
                        else:
                            st.info("No Hold recommendations found.")
                    
                    with col3:
                        st.markdown("### 🔴 **Top Sell Recommendations**")
                        if sell_stocks:
                            # Sort by technical score (lowest first for sell)
                            top_sell = sorted(sell_stocks, key=lambda x: x['technical_score'])[:5]
                            for i, stock in enumerate(top_sell, 1):
                                with st.container():
                                    st.markdown(f"""
                                    **{i}. {stock['ticker']}** - Score: **{stock['technical_score']}/100**
                                    - Price: ${stock['current_price']:.2f}
                                    - RSI: {stock['rsi']:.1f}
                                    - Trend: {'🟢 Bullish' if stock['above_200ma'] else '🔴 Bearish'}
                                    """)
                                    if st.button(f"Analyze {stock['ticker']}", key=f"analyze_sell_{stock['ticker']}"):
                                        st.session_state.current_ticker = stock['ticker']
                                        st.session_state.should_analyze = True
                                        st.rerun()
                        else:
                            st.info("No Sell recommendations found.")
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📊 Screener Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Stocks Analyzed", len(screener_results))
                    with col2:
                        st.metric("Buy Recommendations", len(buy_stocks))
                    with col3:
                        st.metric("Hold Recommendations", len(hold_stocks))
                    with col4:
                        st.metric("Sell Recommendations", len(sell_stocks))
                    
                    # Summary table view
                    st.markdown("---")
                    st.subheader("📋 All Screener Results")
                    
                    # Create summary DataFrame
                    summary_data = []
                    for stock in screener_results:
                        summary_data.append({
                            'Ticker': stock['ticker'],
                            'Score': stock['technical_score'],
                            'Recommendation': stock['recommendation'],
                            'Price': f"${stock['current_price']:.2f}",
                            'RSI': f"{stock['rsi']:.1f}",
                            'Trend': '🟢 Bullish' if stock['above_200ma'] else '🔴 Bearish',
                            'SMA50': f"${stock['sma_50']:.2f}",
                            'SMA200': f"${stock['sma_200']:.2f}",
                            'MACD': f"{stock['macd']:.4f}"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Add color coding to recommendations
                    def color_recommendation(val):
                        if val == 'Buy':
                            return 'background-color: #d4edda; color: #155724'
                        elif val == 'Sell':
                            return 'background-color: #f8d7da; color: #721c24'
                        else:
                            return 'background-color: #fff3cd; color: #856404'
                    
                    # Apply styling and display
                    styled_df = summary_df.style.applymap(color_recommendation, subset=['Recommendation'])
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # Export functionality
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"stock_screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Clear screener results button
                    if st.button("🗑️ Clear Screener Results", key="clear_screener"):
                        st.session_state.screener_results = None
                        st.rerun()
                else:
                    st.info("💡 **Click 'Run Stock Screener' to discover top-performing stocks automatically!**")
                    st.markdown("""
                    **What the screener does:**
                    - Analyzes 40+ popular stocks automatically
                    - Calculates technical scores for each stock
                    - Groups stocks by recommendation category
                    - Shows top 5 performers in each category
                    - Allows quick analysis of any discovered stock
                    
                    **Screening criteria:**
                    - Technical score threshold (adjustable)
                    - RSI, MACD, Moving Averages
                    - Price trends and momentum
                    - Market position indicators
                    """)
        else:
            st.warning("⚠️ Analysis data incomplete. Please analyze a stock again.")
    
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
            if st.button("AAPL", key="btn_aapl"):
                st.session_state.current_ticker = "AAPL"
                st.session_state.should_analyze = True
                st.rerun()
        
        with col2:
            if st.button("MSFT", key="btn_msft"):
                st.session_state.current_ticker = "MSFT"
                st.session_state.should_analyze = True
                st.rerun()
        
        with col3:
            if st.button("GOOGL", key="btn_googl"):
                st.session_state.current_ticker = "GOOGL"
                st.session_state.should_analyze = True
                st.rerun()
        
        with col4:
            if st.button("TSLA", key="btn_tsla"):
                st.session_state.current_ticker = "TSLA"
                st.session_state.should_analyze = True
                st.rerun()

if __name__ == "__main__":
    main()
