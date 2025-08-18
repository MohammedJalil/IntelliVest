#!/usr/bin/env python3
"""
Database Check Script for IntelliVest

This script checks the database connection and verifies if stock data exists.
It can also populate sample data if needed.
"""

import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_database_connection():
    """Create and return a connection to the Supabase database."""
    try:
        # Try connection string first, fall back to individual parameters
        if os.getenv('DATABASE_URL'):
            connection = psycopg2.connect(os.getenv('DATABASE_URL'))
        else:
            connection = psycopg2.connect(
                host=os.getenv('DB_HOST', 'aws-1-us-east-2.pooler.supabase.com'),
                database=os.getenv('DB_NAME', 'postgres'),
                user=os.getenv('DB_USER', 'postgres.yejdhlozdggblspyrure'),
                password=os.getenv('DB_PASS', 'Livescan1!'),
                port=os.getenv('DB_PORT', '6543')
            )
        print("✅ Database connection established successfully")
        return connection
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def check_table_exists(connection):
    """Check if the daily_prices table exists."""
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'daily_prices'
                );
            """)
            exists = cursor.fetchone()[0]
            if exists:
                print("✅ daily_prices table exists")
                return True
            else:
                print("❌ daily_prices table does not exist")
                return False
    except Exception as e:
        print(f"❌ Error checking table: {e}")
        return False

def check_data_exists(connection):
    """Check if any data exists in the daily_prices table."""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM daily_prices;")
            count = cursor.fetchone()[0]
            print(f"📊 Found {count} records in daily_prices table")
            
            if count > 0:
                # Show sample data
                cursor.execute("SELECT DISTINCT ticker FROM daily_prices LIMIT 10;")
                tickers = [row[0] for row in cursor.fetchall()]
                print(f"📈 Sample tickers: {', '.join(tickers)}")
                
                # Show date range
                cursor.execute("SELECT MIN(date), MAX(date) FROM daily_prices;")
                date_range = cursor.fetchone()
                print(f"📅 Date range: {date_range[0]} to {date_range[1]}")
                
                return True
            else:
                print("❌ No data found in daily_prices table")
                return False
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False

def create_table(connection):
    """Create the daily_prices table if it doesn't exist."""
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker VARCHAR(10) NOT NULL,
                    date DATE NOT NULL,
                    open DECIMAL(10,4),
                    high DECIMAL(10,4),
                    low DECIMAL(10,4),
                    close DECIMAL(10,4),
                    volume BIGINT,
                    PRIMARY KEY (ticker, date)
                );
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date 
                ON daily_prices (ticker, date);
            """)
            
            connection.commit()
            print("✅ Table created successfully")
            return True
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        connection.rollback()
        return False

def main():
    """Main function to check database status."""
    print("🔍 Checking IntelliVest database status...")
    print("=" * 50)
    
    # Test connection
    connection = get_database_connection()
    if not connection:
        print("\n❌ Cannot proceed without database connection")
        return
    
    try:
        # Check if table exists
        table_exists = check_table_exists(connection)
        
        if not table_exists:
            print("\n📋 Creating daily_prices table...")
            if create_table(connection):
                print("✅ Table created successfully")
            else:
                print("❌ Failed to create table")
                return
        
        # Check if data exists
        data_exists = check_data_exists(connection)
        
        if not data_exists:
            print("\n🚨 No data found! You need to run the ETL script.")
            print("Run this command to populate data:")
            print("python etl_script.py")
        else:
            print("\n🎉 Database is ready! Your Streamlit app should work now.")
            
    finally:
        connection.close()
        print("\n🔌 Database connection closed")

if __name__ == "__main__":
    main()
