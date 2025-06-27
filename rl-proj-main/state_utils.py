import numpy as np
import pandas as pd


def make_dict_state(price_data):
    MIN_DAYS = 122
    for a in price_data:
        if len(price_data[a]) < MIN_DAYS:
            raise ValueError(
                f"Insufficient data for {a}, need at least {MIN_DAYS} days, got {len(price_data[a])}"
            )

    state = {}
    for a in price_data:
        p = pd.Series(price_data[a])
        n = len(p)

        # returns
        r = p.pct_change()

        # macd
        m_s = p.ewm(span=12, adjust=False).mean()
        m_l = p.ewm(span=26, adjust=False).mean()
        s_p = p.rolling(window=63).std()
        q = (m_s - m_l) / s_p
        s_i = max(62, n - 252)
        s_q = q.iloc[s_i:].std()
        macd = q / s_q if s_q != 0 else np.nan

        # rsi
        N = 30
        g = r.clip(lower=0)
        l = -r.clip(upper=0)
        ag = pd.Series(index=p.index, dtype=float)
        al = pd.Series(index=p.index, dtype=float)
        if n >= N:
            ag.iloc[N - 1] = g.iloc[1 : N + 1].mean()
            al.iloc[N - 1] = l.iloc[1 : N + 1].mean()
            for k in range(N, n):
                ag.iloc[k] = (ag.iloc[k - 1] * (N - 1) + g.iloc[k]) / N
                al.iloc[k] = (al.iloc[k - 1] * (N - 1) + l.iloc[k]) / N
        rsi = pd.Series(index=p.index, dtype=float)
        m = al > 0
        rsi[m] = 100 - 100 / (1 + ag[m] / al[m])
        rsi[~m & (al == 0)] = 100

        state[a] = {
            "prices": p.iloc[-60:].tolist(),
            "returns": r.iloc[-60:].tolist(),
            "MACD": macd.iloc[-60:].tolist(),
            "RSI": rsi.iloc[-60:].tolist(),
        }
        
    # check for NaN values
    for a in state:
        if np.isnan(state[a]["prices"]).any():
            raise ValueError(f"NaN values found in prices for {a}")
        if np.isnan(state[a]["returns"]).any():
            raise ValueError(f"NaN values found in returns for {a}")
        if np.isnan(state[a]["MACD"]).any():
            raise ValueError(f"NaN values found in MACD for {a}")
        if np.isnan(state[a]["RSI"]).any():
            raise ValueError(f"NaN values found in RSI for {a}")

    return state


def make_state(price_data):
    # each asset should be a numpy array of shape (n, 4)
    # where n is the number of days and 4 is the number of features (price, returns, MACD, RSI)
    # then we can stack them together to form a 3D numpy array of shape (m, n, 4)

    dict_state = make_dict_state(price_data)
    n = len(dict_state[list(dict_state.keys())[0]]["prices"])
    m = len(dict_state)
    state = np.zeros((m, n, 4), dtype=float)
    for i, a in enumerate(dict_state):
        state[i, :, 0] = dict_state[a]["prices"]
        state[i, :, 1] = dict_state[a]["returns"]
        state[i, :, 2] = dict_state[a]["MACD"]
        state[i, :, 3] = dict_state[a]["RSI"]

    # check for NaN values
    if np.isnan(state).any():
        raise ValueError("NaN values found in state")
    return state


if __name__ == "__main__":
    n = 150
    price_data = {
        "asset1": np.random.randint(100, 200, size=n).tolist(),
        "asset2": np.random.randint(200, 300, size=n).tolist(),
    }
    dict_state = make_dict_state(price_data)
    for a in dict_state:
        print(f"{a}:")
        print(f"  Prices (last 5): {dict_state[a]['prices'][-5:]}")
        print(f"  Returns (last 5): {dict_state[a]['returns'][-5:]}")
        print(f"  MACD (last 5): {dict_state[a]['MACD'][-5:]}")
        print(f"  RSI (last 5): {dict_state[a]['RSI'][-5:]}")

    state = make_state(price_data)
    print("State shape:", state.shape)
