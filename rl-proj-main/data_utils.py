import pandas as pd
import numpy as np


def preprocess(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # drop columns with any missing values
    df.dropna(axis=1, how="any", inplace=True)

    return df


if __name__ == "__main__":
    file_path = "sp500_data/sp500_stockwise.csv"
    df = preprocess(file_path)

    print(df)

    # count the number of missing values in total
    total_missing = df.isnull().sum().sum()
    print(f"Total missing values: {total_missing}")
