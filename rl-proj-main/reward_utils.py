import numpy as np
import pandas as pd


def reward(cash_t, prices_t, w_t_1, pos_t_1, w_t_2, pos_t_2, bp=0.0001):
    portfolio_value = cash_t + np.dot(w_t_1 * pos_t_1, prices_t)

    pos_change = np.abs(w_t_1 * pos_t_1 - w_t_2 * pos_t_2)
    transaction_costs = np.dot(pos_change, prices_t) * bp
    return portfolio_value, transaction_costs
