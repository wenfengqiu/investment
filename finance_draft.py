import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
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

    # assume the premium received immediately is invested in risk-free assets
    # the max ensures non-negative return when exercised
    # negative return occurs when strike price + premium < current price
    # so "return_if_exercised = 0" when call buyers can do arbitrage
    df["return_if_exercised"] = (
        np.maximum(
            df["adjusted_price"]
            * ((1 + risk_free_rate) ** (df["days_to_expiration"] / 365))
            + df["strike_price"],
            current_price,
        )
        / current_price
    ) ** (365 / df["days_to_expiration"]) - 1

    # drawdown at breakeven
    # decrease of the stock price that yields breakeven from covered call
    # the max ensures no abitrage for covered call seller
    # so "drawdown_at_breakeven" = 1 when arbitrage
    df["drawdown_at_breakeven"] = -(
        (
            (
                np.maximum(
                    current_price
                    - df["adjusted_price"]
                    * ((1 + risk_free_rate) ** (df["days_to_expiration"] / 365)),
                    0,
                )
                / current_price
            )
            ** (365 / df["days_to_expiration"])
        )
        - 1
    )

    return df


def find_outliers(df):
    # currently only label "return_if_exercised"=inf as outliers
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

    df_call = calculate_covered_call_return(
        df=df_clean, current_price=current_price, risk_free_rate=risk_free_rate
    )
    df_call = df_call.loc[df_call["bid"] > 0]
    df_call = df_call.loc[
        (df_call["return_if_exercised"] > risk_free_rate)
        & (df_call["drawdown_at_breakeven"] > desirable_drawdown_rate)
    ]

    df_call = find_outliers(df_call)
    df_call = find_pareto_frontier(
        df_call, "drawdown_at_breakeven", "return_if_exercised"
    )
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H")
    filename = f"{ticker}_{timestamp}.csv"
    df_call.to_csv("reports/" + filename, index=False)
    return df_call


# generate reports
temp = generate_report("RKLB")
plot_scatter_by_pareto(temp, "drawdown_at_breakeven", "return_if_exercised")
