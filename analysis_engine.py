#!/usr/bin/env python3
"""
Analysis Engine for IntelliVest Stock Analysis Project

This module contains the core logic for analyzing stock data, including:
- Technical indicator calculations using ta library
- Technical scoring system based on multiple indicators
- Stock recommendation generation based on scores
"""

import pandas as pd
import ta
from typing import Dict, Any, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to handle both yfinance and standard naming conventions.
    
    Args:
        df (pd.DataFrame): DataFrame with potentially mixed column naming
    
    Returns:
        pd.DataFrame: DataFrame with normalized column names
    """
    # Create a copy to avoid modifying the original
    df_normalized = df.copy()
    
    # Define column name mappings
    column_mappings = {
        'Date': 'date',
        'Open': 'open', 
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mappings.items():
        if old_name in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={old_name: new_name})
    
    return df_normalized


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a given DataFrame of daily prices.
    
    This function uses the ta library to calculate RSI, SMAs, and MACD indicators
    and appends them as new columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data. Must contain columns:
                          'open', 'high', 'low', 'close', 'volume'
    
    Returns:
        pd.DataFrame: Augmented DataFrame with technical indicators added
        
    Raises:
        ValueError: If required columns are missing from the DataFrame
    """
    # Normalize column names first
    df_normalized = normalize_column_names(df)
    
    # Verify required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df_normalized.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(df_normalized.columns)}")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        df_with_indicators = df_normalized.copy()
        
        # Calculate RSI (14-period)
        df_with_indicators['RSI_14'] = ta.momentum.RSIIndicator(df_with_indicators['close'], window=14).rsi()
        
        # Calculate Simple Moving Averages
        df_with_indicators['SMA_50'] = ta.trend.SMAIndicator(df_with_indicators['close'], window=50).sma_indicator()
        df_with_indicators['SMA_200'] = ta.trend.SMAIndicator(df_with_indicators['close'], window=200).sma_indicator()
        
        # Calculate MACD (12, 26, 9)
        macd_indicator = ta.trend.MACD(df_with_indicators['close'], window_slow=26, window_fast=12, window_sign=9)
        df_with_indicators['MACD_12_26_9'] = macd_indicator.macd()
        df_with_indicators['MACDs_12_26_9'] = macd_indicator.macd_signal()
        df_with_indicators['MACDh_12_26_9'] = macd_indicator.macd_diff()
        
        # Remove any rows where indicators couldn't be calculated (NaN values)
        # This typically happens at the beginning of the dataset
        df_with_indicators = df_with_indicators.dropna()
        
        logger.info(f"Successfully calculated technical indicators for {len(df_with_indicators)} data points")
        return df_with_indicators
        
    except Exception as e:
        logger.error(f"Failed to calculate technical indicators: {e}")
        raise


def get_technical_score(latest_indicators: pd.Series) -> int:
    """
    Calculate a technical score from 0-100 based on the latest technical indicators.
    
    The scoring system is rules-based and evaluates multiple technical factors:
    - RSI oversold/overbought conditions
    - Moving average relationships
    - MACD momentum
    - Price relative to long-term moving average
    
    Args:
        latest_indicators (pd.Series): Series containing the most recent row of data
                                      with calculated technical indicators
    
    Returns:
        int: Technical score from 0 to 100
        
    Raises:
        ValueError: If required indicator columns are missing
    """
    # Verify required indicator columns exist
    required_indicators = ['RSI_14', 'SMA_50', 'SMA_200', 'MACD_12_26_9', 
                          'MACDs_12_26_9', 'close']
    missing_indicators = [ind for ind in required_indicators if ind not in latest_indicators.index]
    
    if missing_indicators:
        raise ValueError(f"Missing required indicators: {missing_indicators}")
    
    try:
        score = 0
        
        # RSI Analysis (30 points max)
        rsi = latest_indicators['RSI_14']
        if pd.notna(rsi):  # Check if RSI is not NaN
            if rsi < 30:
                score += 30  # Oversold condition - bullish signal
            elif rsi > 70:
                score -= 10  # Overbought condition - bearish signal
        
        # Moving Average Analysis (30 points max)
        sma_50 = latest_indicators['SMA_50']
        sma_200 = latest_indicators['SMA_200']
        
        if pd.notna(sma_50) and pd.notna(sma_200):
            if sma_50 > sma_200:
                score += 30  # Golden cross - bullish signal
        
        # MACD Analysis (20 points max)
        macd_line = latest_indicators['MACD_12_26_9']
        macd_signal = latest_indicators['MACDs_12_26_9']
        
        if pd.notna(macd_line) and pd.notna(macd_signal):
            if macd_line > macd_signal:
                score += 20  # MACD above signal line - bullish momentum
        
        # Price vs Long-term MA Analysis (20 points max)
        close_price = latest_indicators['close']
        
        if pd.notna(close_price) and pd.notna(sma_200):
            if close_price > sma_200:
                score += 20  # Price above 200-day MA - bullish trend
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        logger.debug(f"Technical score calculated: {score}")
        return score
        
    except Exception as e:
        logger.error(f"Failed to calculate technical score: {e}")
        raise


def get_final_recommendation(final_score: int) -> str:
    """
    Generate a final recommendation string based on the technical score.
    
    The recommendation system uses predefined thresholds to categorize stocks:
    - Strong Buy: Score > 85
    - Buy: Score 70-84
    - Hold: Score 30-69
    - Consider Selling: Score < 30
    
    Args:
        final_score (int): Technical score from 0 to 100
    
    Returns:
        str: Recommendation string
        
    Raises:
        ValueError: If score is outside valid range
    """
    if not isinstance(final_score, (int, float)) or final_score < 0 or final_score > 100:
        raise ValueError("Score must be a number between 0 and 100")
    
    try:
        if final_score > 85:
            recommendation = "Strong Buy"
        elif final_score >= 70:
            recommendation = "Buy"
        elif final_score >= 30:
            recommendation = "Hold"
        else:
            recommendation = "Consider Selling"
        
        logger.info(f"Recommendation generated: {recommendation} (Score: {final_score})")
        return recommendation
        
    except Exception as e:
        logger.error(f"Failed to generate recommendation: {e}")
        raise


def analyze_stock(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete stock analysis function that combines all analysis steps.
    
    This function orchestrates the entire analysis process:
    1. Calculates technical indicators
    2. Generates technical scores for each data point
    3. Provides the latest recommendation
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        Dict[str, Any]: Dictionary containing analysis results:
                       - 'indicators_df': DataFrame with calculated indicators
                       - 'latest_score': Most recent technical score
                       - 'recommendation': Final recommendation string
                       - 'analysis_date': Date of the latest analysis
    """
    try:
        # Calculate technical indicators
        indicators_df = calculate_technical_indicators(df)
        
        if indicators_df.empty:
            raise ValueError("No data available for analysis after calculating indicators")
        
        # Get the latest indicators (most recent row)
        latest_indicators = indicators_df.iloc[-1]
        
        # Calculate technical score
        latest_score = get_technical_score(latest_indicators)
        
        # Generate recommendation
        recommendation = get_final_recommendation(latest_score)
        
        # Get analysis date
        analysis_date = latest_indicators.get('date', pd.Timestamp.now().date())
        
        results = {
            'indicators_df': indicators_df,
            'latest_score': latest_score,
            'recommendation': recommendation,
            'analysis_date': analysis_date
        }
        
        logger.info(f"Stock analysis completed successfully. Score: {latest_score}, "
                   f"Recommendation: {recommendation}")
        
        return results
        
    except Exception as e:
        logger.error(f"Stock analysis failed: {e}")
        raise


def get_historical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical scores for all historical data points.
    
    This function is useful for backtesting and analyzing score trends over time.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with technical indicators and scores for all data points
    """
    try:
        # Calculate technical indicators
        indicators_df = calculate_technical_indicators(df)
        
        if indicators_df.empty:
            raise ValueError("No data available for analysis")
        
        # Calculate scores for each row
        scores = []
        for idx, row in indicators_df.iterrows():
            try:
                score = get_technical_score(row)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Could not calculate score for row {idx}: {e}")
                scores.append(None)
        
        # Add scores column
        indicators_df['technical_score'] = scores
        
        logger.info(f"Historical scores calculated for {len(indicators_df)} data points")
        return indicators_df
        
    except Exception as e:
        logger.error(f"Failed to calculate historical scores: {e}")
        raise
