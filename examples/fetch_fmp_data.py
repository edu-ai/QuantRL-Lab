"""
Example: Fetching data from Financial Modeling Prep (FMP)

FMP provides access to:
- Historical OHLCV (price) data (daily and intraday)
- Analyst grades and ratings
- Company profile information (sector, industry, key metrics)
- Historical sector performance
- Historical industry performance

Requires API key.
"""

from datetime import datetime, timedelta

from quantrl_lab.data.sources import FMPDataSource


def main():
    # Initialize the data loader
    # API key can be provided as argument or via FMP_API_KEY environment variable
    loader = FMPDataSource()

    print("=" * 60)
    print("Financial Modeling Prep (FMP) Data Examples")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Example 1: Fetch historical OHLCV data (daily)
    # ------------------------------------------------------------------
    print("\n[1] Historical OHLCV Data (Daily)")
    print("-" * 40)

    df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start="2024-01-01",
        end="2024-06-01",
        timeframe="1d",  # Daily data
    )
    print(f"Retrieved {len(df)} rows for AAPL")
    print(df.head())

    # ------------------------------------------------------------------
    # Example 2: Fetch intraday data (5-minute bars)
    # ------------------------------------------------------------------
    print("\n[2] Intraday Data (5-minute bars)")
    print("-" * 40)

    # Get last 7 days of intraday data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    df_intraday = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start=start_date,
        end=end_date,
        timeframe="5min",  # 5-minute bars
    )
    print(f"Retrieved {len(df_intraday)} 5-minute bars")
    print(df_intraday.head())

    # ------------------------------------------------------------------
    # Example 3: Different intraday timeframes
    # ------------------------------------------------------------------
    print("\n[3] Different Intraday Timeframes")
    print("-" * 40)

    # FMP supports: 5min, 15min, 30min, 1hour, 4hour
    timeframes = ["15min", "1hour"]

    for tf in timeframes:
        df_tf = loader.get_historical_ohlcv_data(
            symbols="MSFT",
            start=start_date,
            end=end_date,
            timeframe=tf,
        )
        print(f"{tf}: Retrieved {len(df_tf)} bars")

    # ------------------------------------------------------------------
    # Example 4: Fetch analyst grades
    # ------------------------------------------------------------------
    print("\n[4] Historical Analyst Grades")
    print("-" * 40)

    grades = loader.get_historical_grades("AAPL")
    print(f"Retrieved {len(grades)} historical grades")
    print(f"Columns: {grades.columns.tolist()}")
    print(grades.head())

    # ------------------------------------------------------------------
    # Example 5: Fetch analyst ratings
    # ------------------------------------------------------------------
    print("\n[5] Historical Analyst Ratings")
    print("-" * 40)

    ratings = loader.get_historical_rating("AAPL", limit=50)
    print(f"Retrieved {len(ratings)} historical ratings")
    print(f"Columns: {ratings.columns.tolist()}")
    print(ratings.head())

    # ------------------------------------------------------------------
    # Example 6: Fetch company profile
    # ------------------------------------------------------------------
    print("\n[6] Company Profile")
    print("-" * 40)

    profile = loader.get_company_profile("AAPL")
    print(f"Retrieved company profile for {profile.iloc[0]['symbol']}")
    print(f"\nCompany Name: {profile.iloc[0]['companyName']}")
    print(f"Sector: {profile.iloc[0]['sector']}")
    print(f"Industry: {profile.iloc[0]['industry']}")
    print(f"CEO: {profile.iloc[0].get('ceo', 'N/A')}")
    print(f"Market Cap: ${profile.iloc[0].get('mktCap', 0):,.0f}")
    print(f"\nAll columns: {profile.columns.tolist()}")

    # ------------------------------------------------------------------
    # Example 7: Historical sector performance
    # ------------------------------------------------------------------
    print("\n[7] Historical Sector Performance")
    print("-" * 40)

    sector_perf = loader.get_historical_sector_performance("Technology")
    print(f"Retrieved {len(sector_perf)} records for Technology sector")
    print(sector_perf.head())

    # ------------------------------------------------------------------
    # Example 8: Historical industry performance
    # ------------------------------------------------------------------
    print("\n[8] Historical Industry Performance")
    print("-" * 40)

    industry_perf = loader.get_historical_industry_performance("Software")
    print(f"Retrieved {len(industry_perf)} records for Software industry")
    print(industry_perf.head())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
