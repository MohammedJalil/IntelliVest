#!/usr/bin/env python3
"""
Configuration file for IntelliVest Stock Analysis Project

This module centralizes all configuration settings for the project,
including database settings, API configurations, and system parameters.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'intellivest'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASS', ''),
    'port': os.getenv('DB_PORT', '5432')
}

# Data Pipeline Configuration
ETL_CONFIG = {
    'years_of_data': 5,  # Number of years of historical data to download
    'batch_size': 1000,  # Number of records to insert in each batch
    'request_timeout': 30,  # Timeout for web requests in seconds
    'max_retries': 3,  # Maximum number of retries for failed requests
}

# Technical Analysis Configuration
ANALYSIS_CONFIG = {
    'rsi_period': 14,  # RSI calculation period
    'sma_short': 50,   # Short-term SMA period
    'sma_long': 200,   # Long-term SMA period
    'macd_fast': 12,   # MACD fast period
    'macd_slow': 26,   # MACD slow period
    'macd_signal': 9,  # MACD signal period
}

# Scoring System Configuration
SCORING_CONFIG = {
    'rsi_oversold_threshold': 30,    # RSI level considered oversold
    'rsi_overbought_threshold': 70,  # RSI level considered overbought
    'rsi_oversold_points': 30,       # Points awarded for oversold RSI
    'rsi_overbought_penalty': -10,   # Points deducted for overbought RSI
    'sma_golden_cross_points': 30,   # Points for golden cross (SMA50 > SMA200)
    'macd_momentum_points': 20,      # Points for positive MACD momentum
    'price_above_ma_points': 20,     # Points for price above 200-day MA
}

# Trading Strategy Configuration
STRATEGY_CONFIG = {
    'buy_signal_threshold': 70,      # Technical score threshold for buy signals
    'sell_signal_threshold': 30,     # Technical score threshold for sell signals
    'initial_capital': 10000.0,      # Default initial capital for backtesting
    'position_sizing': 'full',       # Position sizing strategy ('full', 'percentage')
    'max_position_size': 1.0,        # Maximum position size as fraction of capital
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': False,
    'log_file': 'intellivest.log',
}

# API Configuration
API_CONFIG = {
    'yfinance_timeout': 30,          # Timeout for Yahoo Finance API calls
    'wikipedia_timeout': 30,         # Timeout for Wikipedia scraping
    'user_agent': 'IntelliVest/1.0 (Educational Project)',  # User agent for web requests
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'chart_dpi': 300,                # DPI for saved charts
    'chart_format': 'png',           # Default chart format
    'enable_plotting': True,         # Whether to display charts
    'save_charts': False,            # Whether to save charts to disk
}

# Validation Functions
def validate_config():
    """
    Validate that all required configuration values are present and valid.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_env_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASS']
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Warning: Missing required environment variables: {missing_vars}")
        print("Please check your .env file configuration.")
        return False
    
    return True

def get_database_url():
    """
    Get the database connection URL for SQLAlchemy (if needed).
    
    Returns:
        str: Database connection URL
    """
    config = DATABASE_CONFIG
    return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

# Configuration validation on import
if __name__ == "__main__":
    print("IntelliVest Configuration:")
    print(f"Database: {DATABASE_CONFIG['database']} on {DATABASE_CONFIG['host']}")
    print(f"ETL Years: {ETL_CONFIG['years_of_data']}")
    print(f"Analysis RSI Period: {ANALYSIS_CONFIG['rsi_period']}")
    print(f"Strategy Buy Threshold: {STRATEGY_CONFIG['buy_signal_threshold']}")
    
    if validate_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has issues")
