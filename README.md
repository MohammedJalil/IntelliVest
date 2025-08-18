# 🚀 IntelliVest - Intelligent Stock Analysis & ML Forecasting Platform

<div align="center">

![IntelliVest Logo](https://img.shields.io/badge/IntelliVest-📈-blue?style=for-the-badge&logo=python)

**Professional Stock Analysis Platform with Machine Learning Price Forecasting**

[![Live Demo](https://img.shields.io/badge/Live_Demo-🌐-green?style=for-the-badge)](https://your-streamlit-app-url.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 🎯 **What is IntelliVest?**

IntelliVest is a **production-ready, enterprise-grade stock analysis platform** that combines traditional technical analysis with cutting-edge machine learning to provide intelligent investment insights. Built with modern data engineering practices, it features automated ETL pipelines, real-time analysis, and predictive forecasting.

### 🌟 **Key Features**
- **🤖 ML-Powered Forecasting**: Facebook Prophet time-series predictions with confidence intervals
- **📊 Professional Technical Analysis**: RSI, MACD, Moving Averages, Volume analysis
- **🚀 Automated Data Pipeline**: GitHub Actions ETL with daily updates
- **🌐 Beautiful Web Interface**: Multi-page Streamlit dashboard with methodology
- **🗄️ Cloud-Native Architecture**: Supabase PostgreSQL with connection pooling
- **📱 Mobile-First Design**: Responsive design for all devices
- **🔬 Educational Content**: Detailed methodology and technical explanations

## 🚀 Features

## 🌐 **Live Demo**

<div align="center">

**[🚀 Try IntelliVest Live!](https://your-streamlit-app-url.streamlit.app)**

*Experience professional stock analysis with ML forecasting in your browser*

</div>

---

## 🚀 Quick Start

### Option 1: Use the Live Web App (Recommended)
Your IntelliVest app is already deployed and working! Simply visit your Streamlit Cloud URL to start analyzing stocks.

### Option 2: Run Locally
If you want to run the app locally or modify it:

- **🌐 Web Application**: Beautiful Streamlit interface accessible from anywhere
- **📊 Interactive Charts**: Candlestick charts with moving averages and technical indicators
- **🤖 AI-Powered Recommendations**: Intelligent scoring system (0-100) with buy/sell/hold signals
- **📈 Technical Analysis**: RSI, SMAs, MACD, and other professional indicators
- **🗄️ Cloud Database**: Supabase PostgreSQL with connection pooling for reliability
- **📱 Responsive Design**: Works perfectly on desktop, tablet, and mobile devices

## 🛠️ **Tech Stack**

<div align="center">

### **Backend & Data Processing**
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue?style=for-the-badge&logo=numpy)
![Prophet](https://img.shields.io/badge/Prophet-1.1+-orange?style=for-the-badge)

### **Web Framework & Visualization**
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.17+-blue?style=for-the-badge&logo=plotly)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue?style=for-the-badge)

### **Database & Infrastructure**
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue?style=for-the-badge&logo=postgresql)
![Supabase](https://img.shields.io/badge/Supabase-Cloud-orange?style=for-the-badge&logo=supabase)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automation-black?style=for-the-badge&logo=github)

### **Financial Data & APIs**
![Yahoo Finance](https://img.shields.io/badge/Yahoo_Finance-API-green?style=for-the-badge)
![Technical Analysis](https://img.shields.io/badge/Technical_Analysis-TA_Lib-blue?style=for-the-badge)

</div>

---

## 📁 Project Structure

```
IntelliVest/
├── 🚀 app.py                    # Main Streamlit web application
├── 📊 pages/                    # Multi-page app structure
│   └── 1_Methodology.py        # Technical methodology & explanations
├── 🤖 ml_forecaster.py         # ML forecasting with Prophet
├── 📈 analysis_engine.py        # Technical analysis & scoring engine
├── 🔄 etl_script.py            # Automated data pipeline
├── 📊 backtester.py            # Strategy backtesting engine
├── 📋 requirements.txt          # Python dependencies
├── 🗄️ .streamlit/              # Streamlit configuration
│   └── secrets.toml            # Database credentials (secure)
├── 🔄 .github/workflows/       # GitHub Actions automation
│   └── etl_pipeline.yml        # Daily ETL automation
├── 🧪 test_app.py              # Simple test suite
└── 📖 README.md                # Project documentation
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
   # Create .env file with your database credentials
   DATABASE_URL=postgresql://username:password@host:port/database
   
   # Or use individual parameters
   DB_HOST=localhost
   DB_NAME=intellivest
   DB_USER=your_username
   DB_PASS=your_password
   DB_PORT=5432
   ```

5. **Set up PostgreSQL database**
   ```sql
   CREATE DATABASE intellivest;
   CREATE USER your_username WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE intellivest TO your_username;
   ```

## 🌐 Web Application

### Live Demo

Your IntelliVest app is now live and accessible from anywhere! Simply:

1. **Enter a stock ticker** (e.g., AAPL, MSFT, GOOGL, TSLA)
2. **Click "🚀 Analyze Stock"**
3. **View interactive charts** and technical analysis
4. **Get AI-powered recommendations** with scoring

### Features

- **📊 Price Charts**: Candlestick charts with 50 & 200-day moving averages
- **📈 Technical Indicators**: RSI, MACD, and moving average analysis
- **🎯 Smart Recommendations**: Buy/Sell/Hold signals based on technical scoring
- **📋 Data Tables**: Recent technical indicator values
- **📱 Responsive Design**: Works perfectly on all devices

## 📊 Backend Usage

### 1. Data Pipeline (ETL)

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