import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from finance_constants import *

dat = yf.Ticker("RKLB")
dat.fast_info["lastPrice"]
# get earnings dates
# dat.get_earnings_dates()


def query_option_chain_data(ticker):
    dat = yf.Ticker(ticker)
    expiration_dates = dat.options
    df = pd.DataFrame()

    for expiration_date in expiration_dates:
        df_calls = dat.option_chain(expiration_date).calls
        df_puts = dat.option_chain(expiration_date).puts
        df = pd.concat([df, df_calls, df_puts], ignore_index=True)

    return df


def clean_option_chain_data(df):
    def parse_occ_contract(symbol):
        ticker = symbol[:-15]
        expiration_date = f"20{symbol[-15:-13]}-{symbol[-13:-11]}-{symbol[-11:-9]}"
        option_type = "call" if symbol[-9] == "C" else "put"
        strike_price = int(symbol[-8:]) / 1000
        return pd.Series([ticker, expiration_date, option_type, strike_price])

    df[["ticker_name", "expiration_date", "option_type", "strike_price"]] = df[
        "contractSymbol"
    ].apply(parse_occ_contract)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], format="%Y-%m-%d")
    df["days_to_expiration"] = (df["expiration_date"] - pd.Timestamp.today()).dt.days
    df["adjusted_price"] = np.minimum(df["lastPrice"], df["bid"])

    return df


def calculate_covered_call_return(df, current_price, risk_free_rate=None):
    df = df.copy()

    if risk_free_rate is None:
        risk_free_rate = RISK_FREE_RATE

    # Assume the premium received immediately is invested in risk-free assets
    df["return_if_exercised"] = (
        np.maximum(
            df["adjusted_price"]
            * ((1 + risk_free_rate) ** (df["days_to_expiration"] / 365))
            + df["strike_price"],
            0,
        )
        / current_price
        - 1
    )

    # Drawdown at breakeven
    df["drawdown_at_breakeven"] = -(
        np.maximum(
            current_price
            - df["adjusted_price"]
            * ((1 + risk_free_rate) ** (df["days_to_expiration"] / 365)),
            0,
        )
        / current_price
    )

    return df


def find_outliers(df):
    df = df.copy()
    df["outlier"] = False
    df.loc[~np.isfinite(df["return_if_exercised"]), "outlier"] = True
    return df


def find_pareto_frontier(df, col_x, col_y):
    df = df.copy()
    df_sorted = df.sort_values(by=[col_x, col_y], ascending=[False, False]).copy()
    df_sorted = df_sorted.loc[~df_sorted["outlier"]]

    pareto_points = []
    max_y_so_far = -np.inf

    for idx, row in df_sorted.iterrows():
        if row[col_y] > max_y_so_far:
            pareto_points.append(idx)
            max_y_so_far = row[col_y]

    df["pareto"] = False
    df.loc[pareto_points, "pareto"] = True

    return df


def plot_scatter_by_pareto(df, col_x, col_y):
    plt.figure(figsize=(8, 10))
    plt.scatter(df[col_x], df[col_y])
    plt.scatter(
        df[df["pareto"]][col_x], df[df["pareto"]][col_y], color="red", alpha=0.5
    )
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.show()


def generate_report(
    ticker,
    risk_free_rate=RISK_FREE_RATE,
    desirable_drawdown_rate=DESIRABLE_DRAWDOWN_RATE,
):
    dat = yf.Ticker(ticker)
    current_price = dat.fast_info["lastPrice"]

    df = query_option_chain_data(ticker)
    df_clean = clean_option_chain_data(df)
    df_clean = df_clean.loc[df_clean["option_type"] == "call"]

    df_call = calculate_covered_call_return(df_clean, current_price, risk_free_rate)
    df_call = df_call.loc[df_call["bid"] > 0]
    df_call = df_call.loc[
        (df_call["return_if_exercised"] > risk_free_rate)
        & (df_call["drawdown_at_breakeven"] > desirable_drawdown_rate)
    ]

    df_call = find_outliers(df_call)
    df_call = find_pareto_frontier(
        df_call, "drawdown_at_breakeven", "return_if_exercised"
    )

    return df_call
