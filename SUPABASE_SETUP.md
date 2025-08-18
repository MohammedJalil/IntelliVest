# Supabase Setup Guide for IntelliVest

## 🚀 Quick Setup (5 minutes)

### Step 1: Create Supabase Account
1. Go to [supabase.com](https://supabase.com)
2. Click "Start your project"
3. Sign up with GitHub (recommended)

### Step 2: Create New Project
1. Click "New Project"
2. Choose your organization
3. Enter project name: `intellivest`
4. Enter database password (save this!)
5. Choose region closest to you
6. Click "Create new project"

### Step 3: Get Database Connection Details
1. Go to **Settings** → **Database**
2. Copy these values:
   - **Host**: `db.your-project-ref.supabase.co`
   - **Database name**: `postgres`
   - **Port**: `5432`
   - **User**: `postgres`
   - **Password**: (the one you created)

### Step 4: Update Streamlit Secrets
Update `.streamlit/secrets.toml`:

```toml
DB_HOST = "db.your-project-ref.supabase.co"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "your_actual_password"
DB_PORT = "5432"
```

### Step 5: Create Tables
Run this SQL in Supabase SQL Editor:

```sql
-- Create the daily_prices table
CREATE TABLE daily_prices (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10,4),
    high DECIMAL(10,4),
    low DECIMAL(10,4),
    close DECIMAL(10,4),
    volume BIGINT,
    PRIMARY KEY (ticker, date)
);

-- Create index for better performance
CREATE INDEX idx_daily_prices_ticker_date ON daily_prices(ticker, date);

-- Enable Row Level Security (RLS)
ALTER TABLE daily_prices ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (for demo purposes)
CREATE POLICY "Allow all operations" ON daily_prices FOR ALL USING (true);
```

### Step 6: Update ETL Script
Update your `etl_script.py` to use the new database connection:

```python
# Replace the connection parameters with your Supabase details
connection = psycopg2.connect(
    host="db.your-project-ref.supabase.co",
    database="postgres",
    user="postgres",
    password="your_password",
    port="5432"
)
```

### Step 7: Run ETL to Populate Data
```bash
python etl_script.py
```

## 🔒 Security Notes

- **Never commit real passwords** to GitHub
- **Use environment variables** for local development
- **Streamlit Cloud** will securely store your secrets
- **Supabase** provides SSL connections by default

## 🌐 Benefits of Supabase

- ✅ **Free tier** (500MB database, 2GB bandwidth)
- ✅ **PostgreSQL compatible** (drop-in replacement)
- ✅ **Automatic backups**
- ✅ **SSL encryption**
- ✅ **Easy scaling**
- ✅ **Great dashboard**

## 🚨 Important

After setting up Supabase, **delete the local `.streamlit/secrets.toml`** file and add it to `.gitignore` to prevent accidentally committing credentials.

## 📱 Next Steps

1. Set up Supabase following this guide
2. Update your secrets
3. Run ETL to populate data
4. Deploy to Streamlit Cloud
5. Test your app!

Your IntelliVest app will then work perfectly on Streamlit Cloud! 🎉
