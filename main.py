#!/usr/bin/env python3
"""
Main entry point for IntelliVest Stock Analysis Project

This script provides a command-line interface to run different components
of the IntelliVest system, including ETL, analysis, and backtesting.
"""

import argparse
import sys
import logging
from datetime import datetime
from typing import Optional

# Import IntelliVest modules
from config import validate_config, STRATEGY_CONFIG
from etl_script import main as run_etl
from analysis_engine import analyze_stock
from backtester import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_analysis(ticker: str, period: str = '2y') -> None:
    """
    Run stock analysis for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for analysis (e.g., '1y', '2y', '5y')
    """
    try:
        import yfinance as yf
        
        logger.info(f"Starting analysis for {ticker}")
        
        # Download stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            logger.error(f"No data found for {ticker}")
            return
        
        # Analyze the stock
        results = analyze_stock(data)
        
        # Display results
        print(f"\n{'='*60}")
        print(f"ANALYSIS RESULTS FOR {ticker.upper()}")
        print(f"{'='*60}")
        print(f"Analysis Date: {results['analysis_date']}")
        print(f"Technical Score: {results['latest_score']}/100")
        print(f"Recommendation: {results['recommendation']}")
        print(f"Data Points Analyzed: {len(results['indicators_df'])}")
        print(f"{'='*60}")
        
        # Show latest technical indicators
        latest = results['indicators_df'].iloc[-1]
        print(f"\nLatest Technical Indicators:")
        print(f"RSI (14): {latest['RSI_14']:.2f}")
        print(f"SMA (50): ${latest['SMA_50']:.2f}")
        print(f"SMA (200): ${latest['SMA_200']:.2f}")
        print(f"MACD: {latest['MACD_12_26_9']:.4f}")
        print(f"MACD Signal: {latest['MACDs_12_26_9']:.4f}")
        print(f"Close Price: ${latest['close']:.2f}")
        
    except ImportError:
        logger.error("yfinance not installed. Please install it with: pip install yfinance")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")


def run_backtest(ticker: str, start_date: str, end_date: str, 
                initial_capital: Optional[float] = None) -> None:
    """
    Run backtesting for a given ticker and date range.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for backtesting (YYYY-MM-DD)
        end_date (str): End date for backtesting (YYYY-MM-DD)
        initial_capital (Optional[float]): Initial capital for backtesting
    """
    try:
        if initial_capital is None:
            initial_capital = STRATEGY_CONFIG['initial_capital']
        
        logger.info(f"Starting backtest for {ticker} from {start_date} to {end_date}")
        
        # Initialize and run backtester
        backtester = Backtester(ticker, start_date, end_date, initial_capital)
        backtester.run_strategy()
        
        # Evaluate performance
        performance = backtester.evaluate_performance()
        
        # Generate charts
        backtester.plot_results()
        
        # Display trade summary
        trades_summary = backtester.get_trade_summary()
        if not trades_summary.empty:
            print("\nTrade Summary:")
            print(trades_summary.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")


def run_etl_pipeline() -> None:
    """Run the ETL pipeline to download and store stock data."""
    try:
        logger.info("Starting ETL pipeline")
        run_etl()
        logger.info("ETL pipeline completed successfully")
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")


def main():
    """Main function to handle command-line arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="IntelliVest - Intelligent Stock Analysis and Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ETL pipeline
  python main.py etl
  
  # Analyze a stock
  python main.py analyze AAPL
  
  # Backtest a strategy
  python main.py backtest MSFT --start 2020-01-01 --end 2023-12-31
  
  # Backtest with custom capital
  python main.py backtest AAPL --start 2021-01-01 --end 2023-12-31 --capital 50000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ETL command
    etl_parser = subparsers.add_parser('etl', help='Run ETL pipeline')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a stock')
    analyze_parser.add_argument('ticker', help='Stock ticker symbol')
    analyze_parser.add_argument('--period', default='2y', 
                               help='Time period for analysis (default: 2y)')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest a trading strategy')
    backtest_parser.add_argument('ticker', help='Stock ticker symbol')
    backtest_parser.add_argument('--start', required=True, 
                                help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', required=True, 
                                help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, 
                                help='Initial capital (default: $10,000)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if configuration is valid
    if not validate_config():
        logger.warning("Configuration validation failed. Some features may not work properly.")
    
    # Execute commands
    if args.command == 'etl':
        run_etl_pipeline()
    elif args.command == 'analyze':
        run_analysis(args.ticker, args.period)
    elif args.command == 'backtest':
        run_backtest(args.ticker, args.start, args.end, args.capital)
    else:
        parser.print_help()
        return
    
    logger.info("Command completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
