#!/usr/bin/env python3
"""
IntelliVest ML Forecaster - Machine Learning Price Prediction Module

This module provides machine learning-based price forecasting using Facebook Prophet
for time series analysis and prediction.
"""

import pandas as pd
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
    """
    def __init__(self, confidence_interval: float = 0.95):
        self.confidence_interval = confidence_interval
        self.model = None
        self.forecast = None
        self.forecast_dates = None

    def prepare_data_for_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            prophet_df = df[['date', 'close']].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            prophet_df = prophet_df.dropna()
            prophet_df = prophet_df.sort_values(by='ds', ascending=True)
            logger.info(f"Prepared {len(prophet_df)} data points for Prophet")
            return prophet_df
        except Exception as e:
            logger.error(f"Error preparing data for Prophet: {e}")
            raise

    def train_model(self, df: pd.DataFrame, changepoint_prior_scale: float = 0.05,
                    seasonality_prior_scale: float = 10.0,
                    holidays_prior_scale: float = 10.0,
                    seasonality_mode: str = 'additive') -> bool:
        try:
            prophet_df = self.prepare_data_for_prophet(df)
            if len(prophet_df) < 30:
                logger.warning("Insufficient data for reliable forecasting (need at least 30 data points)")
                return False
            self.model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                seasonality_mode=seasonality_mode,
                interval_width=self.confidence_interval,
                daily_seasonality='auto',
                weekly_seasonality='auto',
                yearly_seasonality='auto'
            )
            self.model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            self.model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=8
            )
            logger.info("Training Prophet model...")
            self.model.fit(prophet_df)
            logger.info("Prophet model training completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return False

    def generate_forecast(self, periods: int = 30) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model must be trained before generating forecast")
        try:
            future = self.model.make_future_dataframe(periods=periods)
            logger.info(f"Generating {periods}-day forecast...")
            self.forecast = self.model.predict(future)
            self.forecast_dates = self.forecast['ds'].tail(periods)
            logger.info("Forecast generated successfully")
            return self.forecast
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise

    def get_forecast_summary(self) -> Dict[str, Any]:
        if self.forecast is None:
            raise ValueError("No forecast available. Generate forecast first.")
        try:
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
        if self.forecast is None:
            return "Unknown"
        try:
            forecast_df = self.forecast
            if forecast_df is None or len(forecast_df) < 30:
                return "Insufficient Data"
            start_price = float(forecast_df['yhat'].iloc[-30])
            end_price = float(forecast_df['yhat'].iloc[-1])
            change_pct = ((end_price - start_price) / start_price) * 100
            if change_pct > 2:
                return "Bullish (↗️)"
            elif change_pct < -2:
                return "Bearish (↘️)"
            else:
                return "Sideways (→)"
        except Exception as e:
            logger.warning(f"Error calculating trend direction: {e}")
            return "Unknown"

    def _get_volatility_estimate(self) -> str:
        if self.forecast is None:
            return "Unknown"
        try:
            forecast_df = self.forecast
            if forecast_df is None or len(forecast_df) < 7:
                return "Insufficient Data"
            recent_forecast = forecast_df.tail(7)
            yhat_values = recent_forecast['yhat'].astype(float)
            upper_values = recent_forecast['yhat_upper'].astype(float)
            lower_values = recent_forecast['yhat_lower'].astype(float)
            valid_indices = yhat_values != 0
            if not valid_indices.any():
                return "Unknown"
            avg_interval_width = ((upper_values[valid_indices] - lower_values[valid_indices]) / yhat_values[valid_indices]).mean() * 100
            if avg_interval_width > 5:
                return "High Volatility"
            elif avg_interval_width > 2:
                return "Medium Volatility"
            else:
                return "Low Volatility"
        except Exception as e:
            logger.warning(f"Error calculating volatility estimate: {e}")
            return "Unknown"

    def create_forecast_chart(self, historical_df: pd.DataFrame, ticker: str, forecast_days: int = 30) -> go.Figure:
        try:
            if self.forecast is None:
                raise ValueError("No forecast available. Generate forecast first.")

            # Ensure correct types with coercion to handle invalid data
            historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
            historical_df['close'] = pd.to_numeric(historical_df['close'], errors='coerce')
            historical_df['volume'] = pd.to_numeric(historical_df['volume'], errors='coerce')
            self.forecast['ds'] = pd.to_datetime(self.forecast['ds'], errors='coerce')
            self.forecast['yhat'] = pd.to_numeric(self.forecast['yhat'], errors='coerce')

            # Format hover texts
            def safe_hist_hover(row):
                return f"<b>Date</b>: {row['date'].strftime('%Y-%m-%d')}<br><b>Close</b>: ${row['close']:.2f}"

            def safe_forecast_hover(row):
                return f"<b>Date</b>: {row['ds'].strftime('%Y-%m-%d')}<br><b>Forecast</b>: ${row['yhat']:.2f}"

            def safe_vol_hover(row):
                return f"<b>Date</b>: {row['date'].strftime('%Y-%m-%d')}<br><b>Volume</b>: {int(row['volume']):,}"

            historical_df['hover_text'] = historical_df.apply(safe_hist_hover, axis=1)
            self.forecast['hover_text'] = self.forecast.apply(safe_forecast_hover, axis=1)
            historical_df['volume_hover_text'] = historical_df.apply(safe_vol_hover, axis=1)

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{ticker.upper()} Price Forecast', 'Volume'),
                row_heights=[0.7, 0.3]
            )

            fig.add_trace(
                go.Scatter(
                    x=historical_df['date'],
                    y=historical_df['close'],
                    mode='lines',
                    name='Historical Close',
                    line=dict(color='#1f77b4', width=2),
                    hovertext=historical_df['hover_text'],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.forecast['ds'],
                    y=self.forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    hovertext=self.forecast['hover_text'],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.forecast['ds'],
                    y=self.forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.forecast['ds'],
                    y=self.forecast['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    name=f'{self.confidence_interval * 100}% Confidence Interval',
                    showlegend=True,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

            last_historical_date = historical_df['date'].max()
            fig.add_vline(
                x=last_historical_date,
                line_dash="dash",
                line_color="red"
            )
            fig.add_annotation(
                x=last_historical_date,
                y=historical_df['close'].max(),
                text="Forecast Start",
                showarrow=True,
                arrowhead=1
            )
            fig.add_trace(
                go.Bar(
                    x=historical_df['date'],
                    y=historical_df['volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7,
                    hovertext=historical_df['volume_hover_text'],
                    hoverinfo='text'
                ),
                row=2, col=1
            )
            fig.update_layout(
                title=f'{ticker.upper()} Stock Price Forecast ({forecast_days} Days)',
                xaxis_rangeslider_visible=False,
                height=700,
                showlegend=True,
                hovermode='x unified'
            )
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)

            return fig

        except Exception as e:
            logger.error(f"Error creating forecast chart: {e}")
            raise

    def get_model_insights(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        try:
            insights = {
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
    try:
        forecaster = StockForecaster(confidence_interval=confidence_interval)
        if not forecaster.train_model(df):
            raise ValueError("Failed to train forecasting model")
        forecaster.generate_forecast(periods=forecast_days)
        chart = forecaster.create_forecast_chart(df, ticker, forecast_days)
        return forecaster, chart
    except Exception as e:
        logger.error(f"Error in forecast_stock_price: {e}")
        raise

if __name__ == "__main__":
    print("StockForecaster module loaded successfully!")
    print("Use forecast_stock_price() function for quick forecasting.")
