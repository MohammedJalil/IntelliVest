# IntelliVest - Intelligent Stock Analysis and Recommendation System

IntelliVest is a comprehensive Python backend system for stock analysis, technical indicator calculation, and trading strategy backtesting. The system combines data engineering, technical analysis, and quantitative backtesting to provide data-driven investment insights.

## 🚀 Features

- **Automated Data Pipeline**: ETL process for downloading and storing stock data from Yahoo Finance
- **Technical Analysis Engine**: Calculates RSI, SMAs, MACD, and other technical indicators
- **Intelligent Scoring System**: Rules-based scoring algorithm (0-100) for stock evaluation
- **Strategy Backtesting**: Historical performance simulation with buy/sell signals
- **Performance Visualization**: Interactive charts comparing strategy vs. buy-and-hold performance
- **PostgreSQL Integration**: Robust data storage with efficient bulk operations

## 📁 Project Structure

```
IntelliVest/
├── requirements.txt          # Python dependencies
├── env_example.txt          # Environment variables template
├── etl_script.py            # Data extraction, transformation, and loading
├── analysis_engine.py       # Technical analysis and scoring engine
├── backtester.py            # Strategy backtesting and performance evaluation
└── README.md                # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd IntelliVest
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env with your database credentials
   DB_HOST=localhost
   DB_NAME=intellivest
   DB_USER=your_username
   DB_PASS=your_password
   ```

5. **Set up PostgreSQL database**
   ```sql
   CREATE DATABASE intellivest;
   CREATE USER your_username WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE intellivest TO your_username;
   ```

## 📊 Usage

### 1. Data Pipeline (ETL)

Run the ETL script to download and store stock data:

```bash
python etl_script.py
```

This will:
- Scrape S&P 500 ticker symbols from Wikipedia
- Download 5 years of historical data for each ticker
- Store data in PostgreSQL with efficient bulk operations
- Handle errors gracefully and continue processing

### 2. Stock Analysis

Use the analysis engine to evaluate individual stocks:

```python
from analysis_engine import analyze_stock
import yfinance as yf

# Download stock data
stock = yf.Ticker('AAPL')
data = stock.history(period='2y')

# Analyze the stock
results = analyze_stock(data)
print(f"Score: {results['latest_score']}")
print(f"Recommendation: {results['recommendation']}")
```

### 3. Strategy Backtesting

Test trading strategies with historical data:

```python
from backtester import Backtester

# Initialize backtester
backtester = Backtester('MSFT', '2020-01-01', '2023-12-31', 10000)

# Run the strategy
backtester.run_strategy()

# Evaluate performance
performance = backtester.evaluate_performance()

# Generate charts
backtester.plot_results()
```

## 🔧 Technical Details

### Technical Indicators

The system calculates the following indicators:

- **RSI (14-period)**: Relative Strength Index for overbought/oversold conditions
- **SMA (50 & 200)**: Simple Moving Averages for trend analysis
- **MACD (12, 26, 9)**: Moving Average Convergence Divergence for momentum

### Scoring Algorithm

The technical score (0-100) is based on:

- **RSI Analysis** (30 points max):
  - RSI < 30: +30 points (oversold - bullish)
  - RSI > 70: -10 points (overbought - bearish)

- **Moving Averages** (30 points max):
  - SMA_50 > SMA_200: +30 points (golden cross)

- **MACD Momentum** (20 points max):
  - MACD line > Signal line: +20 points

- **Price vs Long-term MA** (20 points max):
  - Close > SMA_200: +20 points

### Trading Strategy

The backtesting strategy implements:

- **Buy Signal**: Technical score crosses above 70
- **Sell Signal**: Technical score crosses below 30
- **Position Management**: Single position at a time
- **Portfolio Tracking**: Real-time equity curve calculation

## 📈 Performance Metrics

The backtesting system provides:

- Final Portfolio Value
- Strategy Return Percentage
- Buy & Hold Return Percentage
- Number of Trades Executed
- Performance vs. Buy & Hold Benchmark

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Database Connection Failures**: Graceful fallback and retry mechanisms
- **Data Download Issues**: Continues processing other tickers
- **Indicator Calculation Errors**: Logs warnings and continues analysis
- **Network Timeouts**: Configurable timeout values for web scraping

## 🔒 Security Considerations

- Database credentials stored in environment variables
- No hardcoded API keys or passwords
- Input validation for all user parameters
- SQL injection protection through parameterized queries

## 📝 Logging

Comprehensive logging throughout the system:

- **INFO**: General operational information
- **WARNING**: Non-critical issues that don't stop execution
- **ERROR**: Critical failures that require attention
- **DEBUG**: Detailed information for troubleshooting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended to provide financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.

## 🆘 Support

For issues and questions:

1. Check the existing issues in the repository
2. Review the logging output for error details
3. Ensure all dependencies are properly installed
4. Verify database connectivity and credentials

## 🔮 Future Enhancements

- Real-time data streaming capabilities
- Machine learning-based scoring algorithms
- Portfolio optimization and risk management
- Web dashboard for interactive analysis
- API endpoints for external integrations
- Additional technical indicators and strategies