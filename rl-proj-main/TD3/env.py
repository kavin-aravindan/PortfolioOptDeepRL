import backtrader as bt
import numpy as np
from collections import deque
from math import floor

from data import load_data


class Strat(bt.Strategy):
    params = (
        ("lb", 60),
        ("limit", 0.2),
        ("agent", None),
    )

    def __init__(self):
        self.n_assets = len(self.datas)
        self.names = [d._name for d in self.datas]

        self.state = {
            name: {
                "prices": deque(maxlen=self.p.lb),
                "returns": deque(maxlen=self.p.lb),
                "MACD": deque(maxlen=self.p.lb),
                "RSI": deque(maxlen=self.p.lb),
            }
            for name in self.names
        }
        # self.state["cash"] = self.broker.getcash()

        self.indicators = {}
        for i, d in enumerate(self.datas):
            name = self.names[i]
            self.indicators[name] = {
                "returns": bt.indicators.PctChange(d.close, period=1),
                "macd": bt.indicators.MACD(
                    d.close, period_me1=12, period_me2=26, period_signal=9
                ),
                "rsi": bt.indicators.RSI(d.close, period=30),
                "price_std": bt.indicators.StdDev(d.close, period=63),
            }

    def next(self):
        # time: t - 1 (we haven't updated the state yet, but backtrader has already updated the indicators)
        if len(self.state[self.names[0]]["prices"]) >= self.p.lb:
            r = self.reward_f()
            self.p.agent.step(self.state, r)

        # time: t
        self.update_state()

        if len(self.state[self.names[0]]["prices"]) >= self.p.lb:
            w = self.p.agent.act(self.state)
            # print(f"Weights: {w}")

            self.w = w
            self.trade()
    
    def reward_f(self):
        # TODO: order failures are not handled yet
        return self.broker.getvalue() - self.prev_portfolio_value

    def update_state(self):
        self.prev_portfolio_value = self.broker.getvalue()

        for i, d in enumerate(self.datas):
            name = self.names[i]

            price = d.close[0]

            # features
            returns = self.indicators[name]["returns"][0]
            macd_raw = self.indicators[name]["macd"].macd[0]
            price_std = self.indicators[name]["price_std"][0]
            macd = macd_raw / price_std if price_std != 0 else np.nan
            rsi = self.indicators[name]["rsi"][0]

            self.state[name]["prices"].append(price)
            self.state[name]["returns"].append(returns)
            self.state[name]["MACD"].append(macd)
            self.state[name]["RSI"].append(rsi)

            # self.state["cash"] = self.broker.getcash()

    def trade(self):
        cash = self.broker.getcash()

        for i in range(self.n_assets):
            d = self.datas[i]

            w = self.w[i]
            pos = np.sign(w)
            w = abs(w)
            
            cur_size = self.getposition(d).size
            price = d.close[0]
            target_cash = cash * self.p.limit * w
            try:
                target_size = floor(target_cash / price) * pos
            except:
                print(f"Error calculating target size for {self.names[i]}: {target_cash}, {price}")
                print(f"Cash: {cash}, Limit: {self.p.limit}, Weight: {w}")
                print("Pos:", pos)
                exit(1)


            if target_size != cur_size:
                if target_size == 0:
                    self.close(d)
                    # self.log(f"Closing {self.names[i]}")
                else:
                    self.order_target_size(d, target_size)
                    # self.log(f"Setting {self.names[i]} to {target_size} shares")
    
    # def notify_order(self, order):
    #     if order.status in [order.Completed]:
    #         self.log(
    #             f"Order done: {order.data._name} - "
    #             f"Size: {order.executed.size}, Price: {order.executed.price}, Commission: {order.executed.comm}"
    #         )
    #     elif order.status in [order.Canceled, order.Margin, order.Rejected]:
    #         self.log(f"Order failed: {order.data._name}, {order.status}")
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")


class Env:
    def __init__(self, data):
        self.data = data

    def simulate(self, agent):
        cerebro = bt.Cerebro()

        for d in self.data:
            cerebro.adddata(d)
        
        cerebro.addstrategy(Strat, agent=agent)

        cerebro.broker.setcash(10_000_000.0)
        cerebro.broker.setcommission(commission=0.001)

        initial_value = cerebro.broker.getvalue()
        cerebro.run()
        final_value = cerebro.broker.getvalue()

        return initial_value, final_value


if __name__ == "__main__":
    class RandomAgent:
        def __init__(self, n_assets):
            self.n_assets = n_assets

        def act(self, state):
            w = np.random.randn(self.n_assets)
            w /= np.sum(np.abs(w))
            return w
        
        def step(self, state, reward):
            pass

    cerebro = bt.Cerebro()

    data = load_data()
    data = data[:5]
    for d in data:
        cerebro.adddata(d)

    agent = RandomAgent(n_assets=len(data))
    cerebro.addstrategy(Strat, agent=agent)

    cerebro.broker.setcash(10_000_000.0)
    cerebro.broker.setcommission(commission=0.001)

    initial_value = cerebro.broker.getvalue()
    cerebro.run()
    final_value = cerebro.broker.getvalue()

    print(f"[{initial_value:.2f}] -> [{final_value:.2f}]")