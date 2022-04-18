import table
import starforce
import numpy as np
import pandas as pd
import random
from datetime import datetime


random.seed(datetime.now())


class StarForceSimulator:
    def __init__(self, start: int, goal: int, item_lv: int, base_price: int = 0,
                 starcatch: bool = False, event51015: bool = False, prevent1216: tuple = (False, False, False, False, False),
                 mvp: str = "bronze", pc_room: bool = False, event30: bool = False):
        self.table = table.StarForceTable(starcatch, prevent1216, event51015).get_modified_table()
        self.interval_cost = starforce.StarForceCost(item_lv, mvp, pc_room, event30, prevent1216).reward_df()
        self.information = pd.merge(self.table, self.interval_cost, left_index=True, right_index=True)
        # self.simulated_cost = self.cumulative_cost(start, goal, base_price)

    def dice(self, state: int) -> str:
        boundary_list = tuple(self.table.loc[state])
        up_top = boundary_list[0]
        keep_top = boundary_list[0] + boundary_list[1]
        down_top = boundary_list[0] + boundary_list[1] + boundary_list[2]

        number = random.random()
        if number < up_top:
            result = 'up'
        elif number < keep_top:
            result = 'keep'
        elif number < down_top:
            result = 'down'
        else:
            result = 'destroy'

        return result

    def cumulative_cost(self, start: int, goal: int, base_price: int = 0) -> int:
        cum_cost = 0
        state = start
        while state < goal:
            try_cost = int(self.interval_cost.loc[state] / 1000)
            starforce_result = self.dice(state)
            if starforce_result == "up":
                state += 1
                cum_cost += try_cost
            elif starforce_result == "keep":
                cum_cost += try_cost
            elif starforce_result == "down":
                state -= 1
                cum_cost += try_cost
            elif starforce_result == "destroy":
                state = 12
                cum_cost += try_cost + base_price
            else:
                ValueError("Something wrong. : cumulative_cost method")
        return cum_cost

    def experimental_mean(self, start: int, goal: int, base_price: int = 0, iteration: int = 1000):
        costs = []
        for _ in range(iteration):
            cost = self.cumulative_cost(start, goal, base_price)
            costs.append(cost)
        return np.mean(costs)

    def realtime_means(self, start: int, goal: int, base_price: int = 0, iteration: int = 1000):
        means = [self.cumulative_cost(start, goal, base_price)]
        for n in range(2, iteration+1):
            mu = (1 - 1/n) * means[-1] + self.cumulative_cost(start, goal, base_price) / n
            means.append(mu)
            print(f"{int(means[-1]):>15,},000")
        return means

    def experiment(self, start: int, goal: int, base_price: int = 0, iteration: int = 1000):
        print(self.realtime_means(start=start, goal=goal, base_price=base_price, iteration=iteration)[-1])
