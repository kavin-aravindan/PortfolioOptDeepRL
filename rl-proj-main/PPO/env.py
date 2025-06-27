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

    def start(self):
        # Called once at the start of the backtest
        self.prev_portfolio_value = self.broker.getvalue()
        # Initialize a variable to store details from the previous step for PPO
        self.ppo_data_from_previous_step = None


    """ def next(self):
        is_current_bar_last = (len(self) == len(self.datas[0]) - 1)
        # print("done", is_current_bar_last)

        if self.ppo_data_from_previous_step is not None:
            reward_at_t = self.reward_f() # reward for action taken at t-1
            done_at_t = is_current_bar_last
            # store the result of the previous step
            self.p.agent.store_transition(
                obs_tensor=self.ppo_data_from_previous_step['obs_tensor'],
                action_tensor=self.ppo_data_from_previous_step['action_tensor'],
                log_prob_old_tensor=self.ppo_data_from_previous_step['log_prob_tensor'],
                reward=reward_at_t,
                value_old_tensor=self.ppo_data_from_previous_step['value_tensor'],
                done=done_at_t
            )
            if done_at_t:
                self.ppo_data_from_previous_step = None

        # now you can get the current state
        self.update_state()

        # time: t - 1 (we haven't updated the state yet, but backtrader has already updated the indicators)
        # if len(self.state[self.names[0]]["prices"]) >= self.p.lb:
        #     r = self.reward_f()
        #     self.p.agent.step(self.state, r)

        # # time: t
        # self.update_state()

        # now we can get the action for the current state
        if len(self.state[self.names[0]]["prices"]) >= self.p.lb and not is_current_bar_last:
            # w = self.p.agent.act(self.state)
            # self.log(f"Weights: {w}")

            current_step_deets = self.p.agent.act_and_get_details(self.state)
            w = current_step_deets['action_numpy']
            self.ppo_data_from_previous_step = current_step_deets

            self.w = w
            self.trade()
        
        elif is_current_bar_last and self.ppo_data_from_previous_step is not None:
            self.ppo_data_from_previous_step = None
     """

    def next(self):

        if self.ppo_data_from_previous_step is not None:
            reward_at_t = self.reward_f() # reward for action taken at t-1
            # store the result of the previous step
            self.p.agent.store_transition(
                obs_tensor=self.ppo_data_from_previous_step['obs_tensor'],
                action_tensor=self.ppo_data_from_previous_step['action_tensor'],
                log_prob_old_tensor=self.ppo_data_from_previous_step['log_prob_tensor'],
                reward=reward_at_t,
                value_old_tensor=self.ppo_data_from_previous_step['value_tensor'],
                done=False
            )

        # now you can get the current state
        self.update_state()

        # time: t - 1 (we haven't updated the state yet, but backtrader has already updated the indicators)
        # if len(self.state[self.names[0]]["prices"]) >= self.p.lb:
        #     r = self.reward_f()
        #     self.p.agent.step(self.state, r)

        # # time: t
        # self.update_state()

        # now we can get the action for the current state
        if len(self.state[self.names[0]]["prices"]) >= self.p.lb:
            # w = self.p.agent.act(self.state)
            # self.log(f"Weights: {w}")

            current_step_deets = self.p.agent.act_and_get_details(self.state)
            w = current_step_deets['action_numpy']
            self.ppo_data_from_previous_step = current_step_deets

            self.w = w
            self.trade()

    def stop(self):
        if self.ppo_data_from_previous_step is not None:
            reward_at_t = self.reward_f()
            # store the result of the previous step
            self.p.agent.store_transition(
                obs_tensor=self.ppo_data_from_previous_step['obs_tensor'],
                action_tensor=self.ppo_data_from_previous_step['action_tensor'],
                log_prob_old_tensor=self.ppo_data_from_previous_step['log_prob_tensor'],
                reward=reward_at_t,
                value_old_tensor=self.ppo_data_from_previous_step['value_tensor'],
                done=True
            
            )
        self.ppo_data_from_previous_step = None

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
            target_size = floor(target_cash / price) * pos

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
