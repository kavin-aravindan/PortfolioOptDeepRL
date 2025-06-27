import backtrader as bt
import numpy as np
import pandas as pd
from collections import defaultdict
from math import floor
from state_utils import make_state


class RLStrat(bt.Strategy):
    params = (("lb", 60), ("n_assets", 3), ("min_data", 122))

    def __init__(self):
        self.hist = defaultdict(list)
        self.names = [d._name for d in self.datas]

        if len(self.names) != self.p.n_assets:
            raise ValueError(
                f"Expected {self.p.n_assets} assets, got {len(self.names)}"
            )

        self.w = np.zeros(self.p.n_assets)
        self.pos = np.zeros(self.p.n_assets, dtype=int)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")

    def next(self):
        for i in range(self.p.n_assets):
            name = self.names[i]
            d = self.datas[i]
            price = d.close[0]
            self.hist[name].append(price)

        if len(self.hist[name]) < self.p.min_data:
            # self.log("Not enough data")
            return

        w, pos = self.rl_algo(self.hist)

        if len(w) != self.p.n_assets or len(pos) != self.p.n_assets:
            self.log("Invalid RL output")
            return

        if not np.isclose(np.sum(w), 1.0, atol=0.1) or np.any(w < 0):
            self.log("Invalid weights")
            return

        if not np.all(np.isin(pos, [-1, 0, 1])):
            self.log("Invalid positions")
            return

        self.log(f"Weights: {w}, Positions: {pos}")
        self.w = w
        self.pos = pos
        self.trade()

    def rl_algo(self, price_data):
        # print(price_data)
        state = make_state(price_data)
        # print(state.shape)
        # print(state)
        w = np.ones(self.p.n_assets) / self.p.n_assets
        pos = np.random.choice([-1, 0, 1], size=self.p.n_assets)
        return w, pos

    def trade(self):
        cash = self.broker.getcash()

        for i in range(self.p.n_assets):
            d = self.datas[i]
            w = self.w[i]
            pos = self.pos[i]
            cur_pos = self.getposition(d).size
            price = d.close[0]
            target_cash = cash * w
            target_size = floor(target_cash / price) * pos

            if target_size != cur_pos:
                if target_size == 0:
                    self.close(d)
                    self.log(f"Closing {self.names[i]}")
                else:
                    self.order_target_size(d, target_size)
                    self.log(f"Setting {self.names[i]} to {target_size} shares")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.log(
                f"Order done: {order.info.get('name', 'Unknown')} - "
                f"Size: {order.executed.size}, Price: {order.executed.price}"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order failed: {order.status}")


class AlwaysLongStrat(bt.Strategy):
    params = (("n_assets", 3), ("cash_fraction", 0.95))

    def __init__(self):
        self.names = [d._name for d in self.datas]

        if len(self.names) != self.p.n_assets:
            raise ValueError(
                f"Expected {self.p.n_assets} assets, got {len(self.names)}"
            )

        self.has_traded = False

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")

    def next(self):
        cash = self.broker.getcash()
        if cash <= 0:
            self.log("No cash available, skipping")
            return

        # Find stocks without positions
        no_position = []
        for i in range(self.p.n_assets):
            d = self.datas[i]
            if self.getposition(d).size == 0:
                no_position.append((d, self.names[i]))

        if not no_position:
            # self.log("All stocks have positions, no further trades")
            return

        # Allocate a fraction of the cash equally among stocks without positions
        cash_to_use = cash * self.p.cash_fraction
        cash_per_asset = cash_to_use / len(no_position)

        for d, name in no_position:
            price = d.close[0]
            size = floor(cash_per_asset / price)
            if size > 0:
                order = self.buy(data=d, size=size, exectype=bt.Order.Market)
                order.addinfo(name=name)
                self.log(f"Buying {size} shares of {name} at price {price:.2f}")
            else:
                self.log(
                    f"Cannot buy {name}: Insufficient cash for {cash_per_asset:.2f}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.log(
                f"Order done: {order.info.get('name', 'Unknown')} - Size: {order.executed.size}, Price: {order.executed.price}, Cash: {self.broker.getcash():.2f}"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(
                f"Order failed: {order.info.get('name', 'Unknown')} - Status: {order.status}"
            )
