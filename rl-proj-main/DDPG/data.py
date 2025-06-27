import pandas as pd
import backtrader as bt
from datetime import datetime
import random


def clean(in_path="data/sp500_raw.csv", out_path="data/sp500.csv"):
    df = pd.read_csv(in_path, index_col=0, parse_dates=True)
    df.dropna(axis=1, how="any", inplace=True)

    df.to_csv(out_path, index=True)
    return df


def load_df(path="data/sp500.csv"):
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df


def load_data(path="data/sp500.csv", fromdate="2000-1-1", todate="2005-1-1", *args, **kwargs):
    data = []
    df = load_df(path)

    if fromdate:
        fromdate = datetime.strptime(fromdate, "%Y-%m-%d")
    if todate:
        todate = datetime.strptime(todate, "%Y-%m-%d")

    if fromdate and todate:
        df = df.loc[fromdate:todate]
    elif fromdate:
        df = df.loc[fromdate:]
    elif todate:
        df = df.loc[:todate]

    tickers = df.columns.get_level_values("Ticker").tolist()

    for t in tickers:
        ticker_df = pd.DataFrame(
            {
                "open": df[("Open", t)].values.ravel(),
                "high": df[("High", t)].values.ravel(),
                "low": df[("Low", t)].values.ravel(),
                "close": df[("Close", t)].values.ravel(),
                "volume": df[("Volume", t)].values.ravel(),
            },
            index=df.index,
        )

        datafeed = bt.feeds.PandasData(
            dataname=ticker_df, 
            name=t, 
            timeframe=bt.TimeFrame.Days, 
            compression=1,
            fromdate=fromdate,
            todate=todate,
            *args, 
            **kwargs
        )

        data.append(datafeed)

    return data


if __name__ == "__main__":
    # clean()

    df = load_df()
    print(df)

    data = load_data()
    print(data[0].p.dataname)
