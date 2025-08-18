#!/usr/bin/env python3
"""
Backtester for IntelliVest Stock Analysis Project

This module implements a backtesting system to validate the trading strategy
by simulating historical performance using technical analysis signals.
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
import logging
from analysis_engine import calculate_technical_indicators, get_technical_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    A class to backtest trading strategies based on technical analysis signals.
    
    This class simulates a trading strategy that:
    - Buys when technical score crosses above 70
    - Sells when technical score crosses below 30
    - Tracks portfolio performance over time
    - Compares strategy performance against buy-and-hold
    """
    
    def __init__(self, ticker: str, start_date: str, end_date: str, initial_capital: float = 10000.0):
        """
        Initialize the backtester with stock data and parameters.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for backtesting (YYYY-MM-DD format)
            end_date (str): End date for backtesting (YYYY-MM-DD format)
            initial_capital (float): Initial portfolio value in dollars
        """
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Portfolio tracking variables
        self.cash = initial_capital
        self.shares = 0
        self.portfolio_value = []
        self.buy_hold_value = []
        self.dates = []
        self.trades = []
        
        # Download and prepare data
        self._download_data()
        self._prepare_data()
        
        logger.info(f"Backtester initialized for {ticker} from {start_date} to {end_date}")
    
    def _download_data(self) -> None:
        """Download historical stock data using yfinance."""
        try:
            logger.info(f"Downloading data for {self.ticker}")
            stock = yf.Ticker(self.ticker)
            self.raw_data = stock.history(start=self.start_date, end=self.end_date)
            
            if self.raw_data.empty:
                raise ValueError(f"No data found for {self.ticker} in the specified date range")
            
            logger.info(f"Downloaded {len(self.raw_data)} data points for {self.ticker}")
            
        except Exception as e:
            logger.error(f"Failed to download data for {self.ticker}: {e}")
            raise
    
    def _prepare_data(self) -> None:
        """Prepare data by calculating technical indicators and scores."""
        try:
            # Reset index to make date a column
            df = self.raw_data.reset_index()
            
            # Rename columns to match analysis engine expectations
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Calculate technical indicators
            self.data = calculate_technical_indicators(df)
            
            # Calculate technical scores for each data point
            scores = []
            for idx, row in self.data.iterrows():
                try:
                    score = get_technical_score(row)
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Could not calculate score for row {idx}: {e}")
                    scores.append(None)
            
            self.data['technical_score'] = scores
            
            # Remove rows with NaN scores (typically at the beginning)
            self.data = self.data.dropna(subset=['technical_score'])
            
            logger.info(f"Data prepared with technical indicators and scores for {len(self.data)} data points")
            
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise
    
    def run_strategy(self) -> None:
        """
        Execute the backtesting strategy.
        
        The strategy rules are:
        - Buy when technical score crosses above 70 (if not already holding)
        - Sell when technical score crosses below 30 (if holding shares)
        - Track portfolio value and buy-and-hold performance
        """
        try:
            logger.info("Starting backtesting strategy execution")
            
            # Initialize tracking variables
            self.cash = self.initial_capital
            self.shares = 0
            self.portfolio_value = []
            self.buy_hold_value = []
            self.dates = []
            self.trades = []
            
            # Get initial price for buy-and-hold calculation
            initial_price = self.data.iloc[0]['close']
            initial_shares_bh = self.initial_capital / initial_price
            
            # Track previous score for signal detection
            prev_score = None
            position = "CASH"  # "CASH" or "LONG"
            
            # Iterate through each data point
            for idx, row in self.data.iterrows():
                current_date = row['date']
                current_price = row['close']
                current_score = row['technical_score']
                
                # Calculate current portfolio values
                current_portfolio_value = self.cash + (self.shares * current_price)
                current_buy_hold_value = initial_shares_bh * current_price
                
                # Store values for plotting
                self.portfolio_value.append(current_portfolio_value)
                self.buy_hold_value.append(current_buy_hold_value)
                self.dates.append(current_date)
                
                # Check for trading signals
                if prev_score is not None:
                    # Buy signal: score crosses above 70 and we're in cash
                    if prev_score <= 70 and current_score > 70 and position == "CASH":
                        # Calculate shares to buy
                        shares_to_buy = self.cash / current_price
                        self.shares = shares_to_buy
                        self.cash = 0
                        position = "LONG"
                        
                        # Record trade
                        trade = {
                            'date': current_date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'portfolio_value': current_portfolio_value,
                            'score': current_score
                        }
                        self.trades.append(trade)
                        
                        logger.info(f"BUY signal at {current_date}: {shares_to_buy:.2f} shares at ${current_price:.2f}")
                    
                    # Sell signal: score crosses below 30 and we're long
                    elif prev_score >= 30 and current_score < 30 and position == "LONG":
                        # Sell all shares
                        self.cash = self.shares * current_price
                        shares_sold = self.shares
                        self.shares = 0
                        position = "CASH"
                        
                        # Record trade
                        trade = {
                            'date': current_date,
                            'action': 'SELL',
                            'price': current_price,
                            'shares': shares_sold,
                            'portfolio_value': current_portfolio_value,
                            'score': current_score
                        }
                        self.trades.append(trade)
                        
                        logger.info(f"SELL signal at {current_date}: {shares_sold:.2f} shares at ${current_price:.2f}")
                
                prev_score = current_score
            
            # Final portfolio value
            final_price = self.data.iloc[-1]['close']
            final_portfolio_value = self.cash + (self.shares * final_price)
            
            logger.info(f"Strategy execution completed. Final portfolio value: ${final_portfolio_value:.2f}")
            
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            raise
    
    def evaluate_performance(self) -> Dict[str, float]:
        """
        Calculate and return performance metrics.
        
        Returns:
            Dict[str, float]: Dictionary containing performance metrics
        """
        try:
            if not self.portfolio_value:
                raise ValueError("No performance data available. Run strategy first.")
            
            # Calculate final values
            final_strategy_value = self.portfolio_value[-1]
            final_buy_hold_value = self.buy_hold_value[-1]
            
            # Calculate returns
            strategy_return = ((final_strategy_value - self.initial_capital) / self.initial_capital) * 100
            buy_hold_return = ((final_buy_hold_value - self.initial_capital) / self.initial_capital) * 100
            
            # Count trades
            num_trades = len(self.trades)
            
            # Performance summary
            performance = {
                'final_portfolio_value': final_strategy_value,
                'strategy_return_pct': strategy_return,
                'buy_hold_return_pct': buy_hold_return,
                'num_trades': num_trades,
                'initial_capital': self.initial_capital
            }
            
            # Print performance summary
            print("\n" + "="*50)
            print(f"BACKTESTING RESULTS FOR {self.ticker}")
            print("="*50)
            print(f"Initial Capital: ${self.initial_capital:,.2f}")
            print(f"Final Portfolio Value: ${final_strategy_value:,.2f}")
            print(f"Total Return (Strategy): {strategy_return:+.2f}%")
            print(f"Total Return (Buy & Hold): {buy_hold_return:+.2f}%")
            print(f"Number of Trades: {num_trades}")
            print(f"Performance vs Buy & Hold: {strategy_return - buy_hold_return:+.2f}%")
            print("="*50)
            
            return performance
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            raise
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Generate and display performance comparison charts.
        
        Args:
            save_path (Optional[str]): Path to save the chart image
        """
        try:
            if not self.portfolio_value:
                raise ValueError("No performance data available. Run strategy first.")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Portfolio Value Comparison
            ax1.plot(self.dates, self.portfolio_value, label='Strategy Portfolio', linewidth=2, color='blue')
            ax1.plot(self.dates, self.buy_hold_value, label='Buy & Hold', linewidth=2, color='red', alpha=0.7)
            ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
            ax1.set_title(f'{self.ticker} - Portfolio Value Comparison', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Technical Scores and Trade Signals
            ax2.plot(self.dates, self.data['technical_score'], label='Technical Score', linewidth=1, color='purple')
            ax2.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Buy Signal (70)')
            ax2.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Sell Signal (30)')
            
            # Mark buy and sell points
            for trade in self.trades:
                if trade['action'] == 'BUY':
                    ax2.scatter(trade['date'], trade['score'], color='green', s=100, marker='^', zorder=5)
                else:  # SELL
                    ax2.scatter(trade['date'], trade['score'], color='red', s=100, marker='v', zorder=5)
            
            ax2.set_title(f'{self.ticker} - Technical Score and Trade Signals', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Technical Score', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Chart saved to {save_path}")
            
            # Display plot
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot results: {e}")
            raise
    
    def get_trade_summary(self) -> pd.DataFrame:
        """
        Return a summary of all trades executed during backtesting.
        
        Returns:
            pd.DataFrame: DataFrame containing trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(self.trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        return trades_df.sort_values('date')


def main():
    """Main execution function for the backtester."""
    try:
        # Example usage: Backtest MSFT from 2020 to present
        ticker = 'MSFT'
        start_date = '2020-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Starting backtest for {ticker}")
        
        # Initialize and run backtester
        backtester = Backtester(ticker, start_date, end_date, initial_capital=10000.0)
        backtester.run_strategy()
        
        # Evaluate performance
        performance = backtester.evaluate_performance()
        
        # Plot results
        backtester.plot_results()
        
        # Display trade summary
        trades_summary = backtester.get_trade_summary()
        if not trades_summary.empty:
            print("\nTrade Summary:")
            print(trades_summary.to_string(index=False))
        
        logger.info("Backtesting completed successfully")
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        raise


if __name__ == "__main__":
    main()
