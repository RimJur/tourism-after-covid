from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import polars as pl
import pyarrow
import yfinance as yf

alt.data_transformers.enable("vegafusion")

# Initialize
df_dataset = pl.read_csv("data.csv")
BENCHMARK_TICKER = "VGWL.DE"
START = "2020-01-01"
START_DATETIME = datetime.strptime(START, "%Y-%m-%d")
END = datetime.now().strftime("%Y-%m-%d")
END_DATETIME = datetime.strptime(END, "%Y-%m-%d")
COVID_END = "2021-01-01"
COVID_END_DATETIME = datetime.strptime(COVID_END, "%Y-%m-%d")
TICKERS = df_dataset["Ticker"].to_list()


# 1. Download and clean data
# 2. Normalize close prices
def download_prices_for_dataset(tickers: list) -> pl.DataFrame:
    df_pd = yf.download(tickers, start=START, end=END, group_by="ticker")
    if not isinstance(df_pd, pd.DataFrame):
        print("Received null data frame from yf.download")
        exit()

    close_columns = [col for col in df_pd.columns if col[1] == "Close"]
    df_pd_close = df_pd[close_columns]
    df_pd_close.columns = [col[0] for col in df_pd_close.columns]
    df = pl.from_pandas(df_pd_close.reset_index())

    df = remove_companies_without_data(df)

    df = df.fill_null(strategy="forward").fill_null(strategy="backward")

    df = aggregate_data_to_3_columns(df)

    df = normalize_stock_prices(df, multiEntities=True)

    return df


# 1. Download and clean data
# 2. Normalize close prices
def download_prices_for_benchmark(ticker: str) -> pl.DataFrame:
    benchmark_pd = yf.download(ticker, start=START, end=END, multi_level_index=False)
    if not isinstance(benchmark_pd, pd.DataFrame):
        print("Received null data frame from yf.download")
        exit()
    benchmark_df = pl.from_pandas(benchmark_pd.reset_index())
    benchmark_df = benchmark_df["Date", "Close"]
    benchmark_df = benchmark_df.with_columns(pl.col("Date").cast(pl.Date))

    date_range = pl.DataFrame(
        {"Date": pl.date_range(START_DATETIME, END_DATETIME, interval="1d", eager=True)}
    )
    date_range = date_range.with_columns(pl.col("Date").cast(pl.Date))

    benchmark_df = date_range.join(benchmark_df, on="Date", how="left")
    benchmark_df = benchmark_df.with_columns(
        pl.col("Close").fill_null(strategy="forward").fill_null(strategy="backward")
    )

    df = normalize_stock_prices(benchmark_df, multiEntities=False)

    return df


def join_region_and_category_to_index_prices(df: pl.DataFrame) -> pl.DataFrame:
    right = df_dataset.select(pl.col("Ticker"), pl.col("Region"), pl.col("Category"))

    df = df.join(right, on="Ticker")

    return df


# Removes companies with no data for first week of dataset
def remove_companies_without_data(df: pl.DataFrame) -> pl.DataFrame:
    # Determine a criteria - no data for first 7 days
    start_date = str(df["Date"].min())
    cutoff_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") + timedelta(days=7)
    first_week = df.filter(pl.col("Date") <= cutoff_date)

    # Filter these companies out
    valid_columns = ["Date"]  # Always keep Date column
    for col in df.columns:
        if col != "Date":
            if first_week[col].drop_nulls().len() > 0:
                valid_columns.append(col)

    df = df.select(valid_columns)

    return df


def aggregate_data_to_3_columns(df: pl.DataFrame) -> pl.DataFrame:
    df = df.unpivot(
        index="Date",
        on=[col for col in df.columns if col != "Date"],
        variable_name="Ticker",
        value_name="Close",
    )
    return df


# Function expects short format data frame
# Besides making starting base a 100 it also fills nulls
def normalize_stock_prices(df: pl.DataFrame, multiEntities: bool) -> pl.DataFrame:
    if multiEntities:
        df = df.with_columns(
            [
                (
                    pl.col("Close")
                    / pl.col("Close").drop_nulls().first().over("Ticker")
                    * 100
                ).alias("Normalized Close")
            ]
        )
    else:
        df = df.with_columns(
            [
                (pl.col("Close") / pl.col("Close").drop_nulls().first() * 100)
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                .alias("Normalized Close")
            ]
        )

    return df


def save_aggregated_chart_of_returns(
    df: pl.DataFrame, aggregate_on: str, file_name: str, start: datetime, end: datetime
) -> None:
    df = df.filter(pl.col("Date").cast(pl.Date).is_between(start, end))
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x="Date:T",
            y="Normalized Close:Q",
            color=alt.Color(f"{aggregate_on}:N"),
        )
    )

    chart.save(f"charts/{file_name}")


def save_index_vs_benchmark_chart(df: pl.DataFrame, file_name: str) -> None:
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x="Date:T", y="Normalized Close:Q", color="Name:N")
    )

    chart.save(f"charts/{file_name}")


# Receive data frames as they are
# Add name column (Benchmark and Index)
def combine_with_benchmark(
    df: pl.DataFrame, df_benchmark: pl.DataFrame
) -> pl.DataFrame:
    df = df.group_by("Date").agg(pl.col("Normalized Close").mean())
    df = df.with_columns(Name=pl.lit("Index"))
    df = df.with_columns(pl.col("Date").cast(pl.Date)).sort("Date", descending=False)

    df_benchmark = df_benchmark.with_columns(Name=pl.lit("Benchmark")).select(
        pl.col("Date"), pl.col("Normalized Close"), pl.col("Name")
    )

    df_combined = pl.concat([df_benchmark, df])

    return df_combined


df_dataset_index = download_prices_for_dataset(TICKERS)

df_benchmark = download_prices_for_benchmark(BENCHMARK_TICKER)

df_combined = combine_with_benchmark(df=df_dataset_index, df_benchmark=df_benchmark)


df_with_cat_and_reg = join_region_and_category_to_index_prices(df_dataset_index)

df_region = (
    df_with_cat_and_reg.group_by(["Date", "Region"])
    .agg(pl.col("Normalized Close").mean())
    .sort("Date")
)
df_category = (
    df_with_cat_and_reg.group_by(["Date", "Category"])
    .agg(pl.col("Normalized Close").mean())
    .sort("Date")
)

# PHASE 1 - distributions

# Prepare data with counts and percentages for charts
category_counts = (
    df_with_cat_and_reg.unique(subset=["Ticker"])
    .group_by("Category")
    .agg(pl.len().alias("Count"))
)
total_category = category_counts["Count"].sum()
category_counts = category_counts.with_columns(
    ((pl.col("Count") / total_category) * 100).round(1).alias("Percentage")
).with_columns(
    (pl.col("Count").cast(str) + " (" + pl.col("Percentage").cast(str) + "%)").alias(
        "Label"
    )
)

region_counts = (
    df_with_cat_and_reg.unique(subset=["Ticker"])
    .group_by("Region")
    .agg(pl.len().alias("Count"))
)
total_region = region_counts["Count"].sum()
region_counts = region_counts.with_columns(
    ((pl.col("Count") / total_region) * 100).round(1).alias("Percentage")
).with_columns(
    (pl.col("Count").cast(str) + " (" + pl.col("Percentage").cast(str) + "%)").alias(
        "Label"
    )
)

# Print summary statistics
print("\n=== CATEGORY DISTRIBUTION ===")
for row in category_counts.iter_rows(named=True):
    print(f"{row['Category']}: {row['Count']} ({row['Percentage']}%)")

print("\n=== REGION DISTRIBUTION ===")
for row in region_counts.iter_rows(named=True):
    print(f"{row['Region']}: {row['Count']} ({row['Percentage']}%)")

# Chart 1 - category distribution
chart1 = (
    alt.Chart(category_counts.to_pandas())
    .mark_arc(innerRadius=50, outerRadius=120)
    .encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color("Category:N"),
        tooltip=["Category:N", "Count:Q", "Percentage:Q"],
    )
) + (
    alt.Chart(category_counts.to_pandas())
    .mark_text(radius=140, fontSize=12, fontWeight="bold")
    .encode(theta=alt.Theta("Count:Q"), text="Label:N", color=alt.value("black"))
)
chart1.save("charts/category-distribution.png")

# Chart 2 - region distribution
chart2 = (
    alt.Chart(region_counts.to_pandas())
    .mark_arc(innerRadius=50, outerRadius=120)
    .encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color("Region:N"),
        tooltip=["Region:N", "Count:Q", "Percentage:Q"],
    )
) + (
    alt.Chart(region_counts.to_pandas())
    .mark_text(radius=140, fontSize=12, fontWeight="bold")
    .encode(theta=alt.Theta("Count:Q"), text="Label:N", color=alt.value("black"))
)
chart2.save("charts/region-distribution.png")

# Chart 3 - market cap distribution (period end)

# PHASE 2 - results

# Chart 4 - index vs. benchmark
save_index_vs_benchmark_chart(df_combined, "index-vs-benchmark-full-timeframe.png")


# Chart 5 - by ticker full timeframe - CAGR Analysis
def calculate_ticker_cagr_statistics(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Calculate CAGR returns and statistics for all tickers"""
    df_filtered = df.filter(
        pl.col("Date").cast(pl.Date).is_between(START_DATETIME, END_DATETIME)
    )

    # Calculate number of years
    years = (END_DATETIME - START_DATETIME).days / 365.25

    # Calculate CAGR for each ticker
    cagr_stats = (
        df_filtered.group_by("Ticker")
        .agg(
            [
                pl.col("Normalized Close").first().alias("Initial_Value"),
                pl.col("Normalized Close").last().alias("Final_Value"),
            ]
        )
        .with_columns(
            [
                # CAGR = (Ending Value / Beginning Value)^(1/years) - 1
                (
                    (pl.col("Final_Value") / pl.col("Initial_Value")) ** (1 / years) - 1
                ).alias("CAGR")
            ]
        )
        .sort("CAGR", descending=True)
    )

    # Calculate summary statistics for CAGR
    cagr_values = cagr_stats["CAGR"]
    summary_stats = {
        "mean": cagr_values.mean(),
        "median": cagr_values.median(),
        "std": cagr_values.std(),
        "count": len(cagr_values),
    }

    return cagr_stats, summary_stats


# Generate CAGR analysis
ticker_cagr, cagr_summary = calculate_ticker_cagr_statistics(df_dataset_index)

print("\n=== CAGR ANALYSIS SUMMARY ===")
print(f"Mean CAGR: {cagr_summary['mean'] * 100:.2f}%")
print(f"Median CAGR: {cagr_summary['median'] * 100:.2f}%")
print(f"Standard Deviation: {cagr_summary['std'] * 100:.2f}%")

print("\n=== TOP 5 CAGR PERFORMERS ===")
top_5 = ticker_cagr.head(5)
for row in top_5.iter_rows(named=True):
    print(f"{row['Ticker']}: {row['CAGR'] * 100:.2f}%")

print("\n=== BOTTOM 5 CAGR PERFORMERS ===")
bottom_5 = ticker_cagr.tail(5).reverse()
for row in bottom_5.iter_rows(named=True):
    print(f"{row['Ticker']}: {row['CAGR'] * 100:.2f}%")

# Save detailed CAGR data to CSV
ticker_cagr.write_csv("charts/ticker-cagr-analysis.csv")


# Chart 6 - by performance full timeframe
save_aggregated_chart_of_returns(
    df_region,
    "Region",
    "performance-by-region-full-timeframe.png",
    START_DATETIME,
    END_DATETIME,
)
# Chart 7 - by category full timeframe
save_aggregated_chart_of_returns(
    df_category,
    "Category",
    "performance-by-category-full-timeframe.png",
    START_DATETIME,
    END_DATETIME,
)
# Chart 8 - by region covid timeframe
save_aggregated_chart_of_returns(
    df_region,
    "Region",
    "performance-by-region-covid-timeframe.png",
    START_DATETIME,
    COVID_END_DATETIME,
)
# Chart 9 - by category covid timeframe
save_aggregated_chart_of_returns(
    df_category,
    "Category",
    "performance-by-category-covid-timeframe.png",
    START_DATETIME,
    COVID_END_DATETIME,
)

# Chart 10 - by region vs. benchmark covid timeframe

# Chart 11 - by category vs. benchmark covid timeframe
