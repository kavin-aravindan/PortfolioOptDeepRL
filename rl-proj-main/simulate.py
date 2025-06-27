import datetime
import backtrader as bt
import numpy as np
import pandas as pd

from strategies import RLStrat as Strat

# from strategies import AlwaysLongStrat as Strat


def main():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strat, n_assets=3)

    for i in range(3):
        np.random.seed(i)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        data = pd.DataFrame(
            {
                "open": 100 + np.random.randn(len(dates)).cumsum(),
                "high": 100 + np.random.randn(len(dates)).cumsum() + 1,
                "low": 100 + np.random.randn(len(dates)).cumsum() - 1,
                "close": 100 + np.random.randn(len(dates)).cumsum(),
                "volume": 1000,
            },
            index=dates,
        )
        feed = bt.feeds.PandasData(dataname=data, name=f"Asset_{i+1}")
        cerebro.adddata(feed)

    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0001)

    print(f"Start Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()

    print(f"End Value: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()


if __name__ == "__main__":
    main()
