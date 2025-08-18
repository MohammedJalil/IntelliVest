#!/usr/bin/env python3
"""
IntelliVest ML Forecaster - Machine Learning Price Prediction Module

This module provides machine learning-based price forecasting using Facebook Prophet
for time series analysis and prediction.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Tuple, Dict, Any
import logging

# Suppress Prophet warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockForecaster:
    """
    A class for forecasting stock prices using Facebook Prophet.
    
    Prophet is excellent for time series forecasting because it:
    - Handles missing data gracefully
    - Automatically detects seasonality
    - Provides uncertainty intervals
    - Is robust to outliers
    """
    
    def __init__(self, confidence_interval: float = 0.95):
        """
        Initialize the forecaster.
        
        Args:
            confidence_interval (float): Confidence interval for predictions (0.95 = 95%)
        """
        self.confidence_interval = confidence_interval
        self.model = None
        self.forecast = None
        self.forecast_dates = None
        
    def prepare_data_for_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare stock data for Prophet model.
        
        Prophet requires columns named 'ds' (date) and 'y' (value).
        
        Args:
            df (pd.DataFrame): DataFrame with 'date' and 'close' columns
            
        Returns:
            pd.DataFrame: DataFrame formatted for Prophet
        """
        try:
            # Create a copy to avoid modifying original data
            prophet_df = df[['date', 'close']].copy()
            
            # Rename columns for Prophet
            prophet_df.columns = ['ds', 'y']
            
            # Ensure date is datetime
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Remove any rows with missing values
            prophet_df = prophet_df.dropna()
            
            # Sort by date
            prophet_df = prophet_df.sort_values('ds')
            
            logger.info(f"Prepared {len(prophet_df)} data points for Prophet")
            return prophet_df
            
        except Exception as e:
            logger.error(f"Error preparing data for Prophet: {e}")
            raise
    
    def train_model(self, df: pd.DataFrame, 
                   changepoint_prior_scale: float = 0.05,
                   seasonality_prior_scale: float = 10.0,
                   holidays_prior_scale: float = 10.0,
                   seasonality_mode: str = 'additive') -> bool:
        """
        Train the Prophet model on historical stock data.
        
        Args:
            df (pd.DataFrame): Historical stock data
            changepoint_prior_scale (float): Flexibility of the trend (higher = more flexible)
            seasonality_prior_scale (float): Flexibility of the seasonality
            holidays_prior_scale (float): Flexibility of the holiday effects
            seasonality_mode (str): 'additive' or 'multiplicative'
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Prepare data
            prophet_df = self.prepare_data_for_prophet(df)
            
            if len(prophet_df) < 30:
                logger.warning("Insufficient data for reliable forecasting (need at least 30 data points)")
                return False
            
            # Initialize Prophet model with custom parameters
            self.model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                seasonality_mode=seasonality_mode,
                interval_width=self.confidence_interval,
                daily_seasonality=False,  # Disable for daily stock data
                weekly_seasonality=True,  # Enable weekly patterns
                yearly_seasonality=True   # Enable yearly patterns
            )
            
            # Add custom seasonality for stock market patterns
            self.model.add_seasonality(
                name='monthly', 
                period=30.5, 
                fourier_order=5
            )
            
            # Add quarterly seasonality (earnings seasons)
            self.model.add_seasonality(
                name='quarterly', 
                period=91.25, 
                fourier_order=8
            )
            
            # Fit the model
            logger.info("Training Prophet model...")
            self.model.fit(prophet_df)
            logger.info("Prophet model training completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return False
    
    def generate_forecast(self, periods: int = 30) -> pd.DataFrame:
        """
        Generate price forecast for the specified number of periods.
        
        Args:
            periods (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: Forecast data with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating forecast")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            
            # Generate forecast
            logger.info(f"Generating {periods}-day forecast...")
            self.forecast = self.model.predict(future)
            
            # Store forecast dates for easy access
            self.forecast_dates = self.forecast['ds'].tail(periods)
            
            logger.info("Forecast generated successfully")
            return self.forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the forecast results.
        
        Returns:
            Dict[str, Any]: Summary statistics and predictions
        """
        if self.forecast is None:
            raise ValueError("No forecast available. Generate forecast first.")
        
        try:
            # Get the last few predictions
            recent_forecast = self.forecast.tail(7)
            
            summary = {
                'next_7_days': {
                    'dates': recent_forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'predictions': recent_forecast['yhat'].round(2).tolist(),
                    'lower_bound': recent_forecast['yhat_lower'].round(2).tolist(),
                    'upper_bound': recent_forecast['yhat_upper'].round(2).tolist()
                },
                'confidence_interval': f"{self.confidence_interval * 100}%",
                'trend_direction': self._get_trend_direction(),
                'volatility_estimate': self._get_volatility_estimate()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting forecast summary: {e}")
            raise
    
    def _get_trend_direction(self) -> str:
        """Determine the overall trend direction of the forecast."""
        if self.forecast is None:
            return "Unknown"
        
        try:
            # Compare start and end of forecast period
            start_price = self.forecast['yhat'].iloc[-30]  # 30 days ago
            end_price = self.forecast['yhat'].iloc[-1]     # Latest prediction
            
            change_pct = ((end_price - start_price) / start_price) * 100
            
            if change_pct > 2:
                return "Bullish (↗️)"
            elif change_pct < -2:
                return "Bearish (↘️)"
            else:
                return "Sideways (→)"
                
        except Exception:
            return "Unknown"
    
    def _get_volatility_estimate(self) -> str:
        """Estimate volatility based on confidence intervals."""
        if self.forecast is None:
            return "Unknown"
        
        try:
            # Calculate average width of confidence intervals
            recent_forecast = self.forecast.tail(7)
            avg_interval_width = (
                (recent_forecast['yhat_upper'] - recent_forecast['yhat_lower']) / 
                recent_forecast['yhat']
            ).mean() * 100
            
            if avg_interval_width > 5:
                return "High Volatility"
            elif avg_interval_width > 2:
                return "Medium Volatility"
            else:
                return "Low Volatility"
                
        except Exception:
            return "Unknown"
    
    def create_forecast_chart(self, historical_df: pd.DataFrame, 
                             ticker: str, 
                             forecast_days: int = 30) -> go.Figure:
        """
        Create a comprehensive forecast chart showing historical data and predictions.
        
        Args:
            historical_df (pd.DataFrame): Historical stock data
            ticker (str): Stock ticker symbol
            forecast_days (int): Number of days to forecast
            
        Returns:
            go.Figure: Plotly figure with historical data and forecast
        """
        try:
            if self.forecast is None:
                raise ValueError("No forecast available. Generate forecast first.")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{ticker.upper()} Price Forecast', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=historical_df['date'],
                    y=historical_df['close'],
                    mode='lines',
                    name='Historical Close',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Forecast line
            fig.add_trace(
                go.Scatter(
                    x=self.forecast['ds'],
                    y=self.forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    hovertemplate='<b>%{x}</b><br>Forecast: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=self.forecast['ds'].tolist() + self.forecast['ds'].tolist()[::-1],
                    y=self.forecast['yhat_upper'].tolist() + self.forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255, 127, 14, 0)'),
                    name=f'{self.confidence_interval * 100}% Confidence Interval',
                    showlegend=True,
                    hovertemplate='<b>%{x}</b><br>Upper: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add vertical line to separate historical from forecast
            last_historical_date = historical_df['date'].max()
            fig.add_vline(
                x=last_historical_date,
                line_dash="dash",
                line_color="red",
                annotation_text="Forecast Start",
                annotation_position="top right"
            )
            
            # Volume chart (historical only)
            fig.add_trace(
                go.Bar(
                    x=historical_df['date'],
                    y=historical_df['volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7,
                    hovertemplate='<b>%{x}</b><br>Volume: %{y:,}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{ticker.upper()} Stock Price Forecast ({forecast_days} Days)',
                xaxis_rangeslider_visible=False,
                height=700,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating forecast chart: {e}")
            raise
    
    def get_model_insights(self) -> Dict[str, Any]:
        """
        Get insights about the trained model.
        
        Returns:
            Dict[str, Any]: Model insights and diagnostics
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        try:
            # Get model components
            components = self.model.plot_components(self.forecast)
            
            insights = {
                'trend_strength': self._analyze_trend_strength(),
                'seasonality_detected': self._check_seasonality(),
                'changepoints': len(self.model.changepoints) if hasattr(self.model, 'changepoints') else 0,
                'model_parameters': {
                    'changepoint_prior_scale': self.model.changepoint_prior_scale,
                    'seasonality_prior_scale': self.model.seasonality_prior_scale,
                    'holidays_prior_scale': self.model.holidays_prior_scale
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting model insights: {e}")
            return {}

def forecast_stock_price(df: pd.DataFrame, 
                        ticker: str, 
                        forecast_days: int = 30,
                        confidence_interval: float = 0.95) -> Tuple[StockForecaster, go.Figure]:
    """
    Convenience function to forecast stock prices.
    
    Args:
        df (pd.DataFrame): Historical stock data
        ticker (str): Stock ticker symbol
        forecast_days (int): Number of days to forecast
        confidence_interval (float): Confidence interval for predictions
        
    Returns:
        Tuple[StockForecaster, go.Figure]: Forecaster object and forecast chart
    """
    try:
        # Initialize forecaster
        forecaster = StockForecaster(confidence_interval=confidence_interval)
        
        # Train model
        if not forecaster.train_model(df):
            raise ValueError("Failed to train forecasting model")
        
        # Generate forecast
        forecaster.generate_forecast(periods=forecast_days)
        
        # Create chart
        chart = forecaster.create_forecast_chart(df, ticker, forecast_days)
        
        return forecaster, chart
        
    except Exception as e:
        logger.error(f"Error in forecast_stock_price: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    print("StockForecaster module loaded successfully!")
    print("Use forecast_stock_price() function for quick forecasting.")
