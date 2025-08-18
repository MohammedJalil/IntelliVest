#!/usr/bin/env python3
"""
ETL Script for IntelliVest Stock Analysis Project

This script performs the Extract, Transform, Load (ETL) process for stock data.
It scrapes S&P 500 tickers, downloads historical price data, and loads it into a PostgreSQL database.

Designed to be run on a schedule (e.g., daily) to maintain up-to-date stock data.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_database_connection() -> psycopg2.extensions.connection:
    """
    Create and return a connection to the PostgreSQL database.
    
    Returns:
        psycopg2.extensions.connection: Database connection object
        
    Raises:
        psycopg2.Error: If connection fails
    """
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS')
        )
        logger.info("Database connection established successfully")
        return connection
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def create_tables(connection: psycopg2.extensions.connection) -> None:
    """
    Create the necessary database tables if they don't exist.
    
    Args:
        connection (psycopg2.extensions.connection): Database connection object
    """
    try:
        with connection.cursor() as cursor:
            # Create daily_prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open NUMERIC(10, 4) NOT NULL,
                    high NUMERIC(10, 4) NOT NULL,
                    low NUMERIC(10, 4) NOT NULL,
                    close NUMERIC(10, 4) NOT NULL,
                    volume BIGINT NOT NULL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            
            # Create index for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date 
                ON daily_prices (ticker, date)
            """)
            
            connection.commit()
            logger.info("Database tables created/verified successfully")
            
    except psycopg2.Error as e:
        logger.error(f"Failed to create tables: {e}")
        connection.rollback()
        raise


def get_sp500_tickers() -> List[str]:
    """
    Scrape S&P 500 ticker symbols from Wikipedia.
    
    Returns:
        List[str]: List of S&P 500 ticker symbols
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the first table (S&P 500 companies table)
        table = soup.find('table', {'class': 'wikitable'})
        if not table:
            raise ValueError("Could not find S&P 500 companies table on Wikipedia")
        
        tickers = []
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 1:
                ticker = cells[0].text.strip()
                if ticker:  # Ensure ticker is not empty
                    tickers.append(ticker)
        
        logger.info(f"Successfully scraped {len(tickers)} S&P 500 tickers")
        return tickers
        
    except Exception as e:
        logger.error(f"Failed to scrape S&P 500 tickers: {e}")
        # Return a small subset of major tickers as fallback
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B']
        logger.info(f"Using fallback tickers: {fallback_tickers}")
        return fallback_tickers


def download_stock_data(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Download historical stock data for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        years (int): Number of years of historical data to download
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        
        # Reset index to make date a column
        data = data.reset_index()
        
        # Rename columns to match database schema
        data = data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add ticker column
        data['ticker'] = ticker
        
        # Select only the columns we need
        data = data[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Convert date to date type (remove time component)
        data['date'] = pd.to_datetime(data['date']).dt.date
        
        logger.info(f"Downloaded {len(data)} records for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to download data for {ticker}: {e}")
        return pd.DataFrame()


def insert_stock_data(connection: psycopg2.extensions.connection, data: pd.DataFrame) -> None:
    """
    Insert stock data into the database using bulk insert.
    
    Args:
        connection (psycopg2.extensions.connection): Database connection object
        data (pd.DataFrame): DataFrame containing stock data to insert
    """
    if data.empty:
        return
    
    try:
        with connection.cursor() as cursor:
            # Prepare data for bulk insert
            records = data.to_dict('records')
            
            # Use execute_values for efficient bulk insert
            execute_values(
                cursor,
                """
                INSERT INTO daily_prices (ticker, date, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (ticker, date) 
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                [(r['ticker'], r['date'], r['open'], r['high'], r['low'], r['close'], r['volume']) 
                 for r in records]
            )
            
            connection.commit()
            logger.info(f"Successfully inserted/updated {len(records)} records")
            
    except psycopg2.Error as e:
        logger.error(f"Failed to insert data: {e}")
        connection.rollback()
        raise


def main() -> None:
    """
    Main ETL function that orchestrates the entire data pipeline.
    """
    logger.info("Starting IntelliVest ETL process")
    
    try:
        # Get database connection
        connection = get_database_connection()
        
        # Create tables if they don't exist
        create_tables(connection)
        
        # Get S&P 500 tickers
        tickers = get_sp500_tickers()
        logger.info(f"Processing {len(tickers)} tickers")
        
        # Process each ticker
        successful_tickers = 0
        failed_tickers = 0
        
        for i, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"Processing ticker {i}/{len(tickers)}: {ticker}")
                
                # Download stock data
                data = download_stock_data(ticker)
                
                if not data.empty:
                    # Insert data into database
                    insert_stock_data(connection, data)
                    successful_tickers += 1
                else:
                    failed_tickers += 1
                    
            except Exception as e:
                logger.error(f"Failed to process ticker {ticker}: {e}")
                failed_tickers += 1
                continue
        
        # Log summary
        logger.info(f"ETL process completed. Successful: {successful_tickers}, Failed: {failed_tickers}")
        
    except Exception as e:
        logger.error(f"ETL process failed: {e}")
        raise
        
    finally:
        if 'connection' in locals():
            connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()
