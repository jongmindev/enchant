import table
import starforce
import calculator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

from datetime import datetime


random.seed(datetime.now())


class StarForceSimulator:
    def __init__(self, start: int, goal: int, item_lv: int, base_price: int = 0,
                 starcatch: bool = False, event51015: bool = False, prevent1216: tuple = (False, False, False, False, False),
                 mvp: str = "bronze", pc_room: bool = False, event30: bool = False):
        self.table = table.StarForceTable(starcatch, prevent1216, event51015).get_modified_table()
        self.interval_cost = starforce.StarForceCost(item_lv, mvp, pc_room, event30, prevent1216).reward_df()
        self.information = pd.merge(self.table, self.interval_cost, left_index=True, right_index=True)

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


class StarforceSimulatorVer2:
    def __init__(self, start: int, goal: int, item_lv: int, base_price: int = 0,
                 starcatch: bool = False, event51015: bool = False,
                 prevent1216: tuple = (False, False, False, False, False),
                 mvp: str = "bronze", pc_room: bool = False, event30: bool = False, event1plus1: bool = False):
        self.table = table.StarForceTable(starcatch, prevent1216, event51015).prob_table
        self.interval_cost = starforce.StarForceCost(item_lv, mvp, pc_room, event30, prevent1216).reward_df()
        self.information = pd.merge(self.table, self.interval_cost, left_index=True, right_index=True)
        self.reduced_cost = (self.interval_cost / 1000).astype(int)
        self.start, self.goal, self.base_price = start, goal, base_price
        self.simulated_reduced_cost = None
        self.markov_class = calculator.StarForceCalculator(goal, item_lv, base_price, starcatch, mvp, pc_room,
                                                           event30, event51015, event1plus1, prevent1216)

    def cumulative_cost(self) -> int:
        try_costs = np.array(self.reduced_cost["interval_cost"])
        probability_table = np.array(self.table)

        cum_cost = 0
        state = self.start

        while state < self.goal:
            boundaries = probability_table[state]
            up_top = boundaries[0]
            keep_top = up_top + boundaries[1]
            down_top = keep_top + boundaries[2]

            dice = random.random()
            try_cost = try_costs[state]

            # 스타포스 성공
            if dice < up_top:
                state += 1
                cum_cost += try_cost
            # 스타포스 유지
            elif dice < keep_top:
                cum_cost += try_cost
            # 스타포스 하락
            elif dice < down_top:
                state -= 1
                cum_cost += try_cost
            # 파괴
            else:
                state = 12
                cum_cost += try_cost + self.base_price / 1000

        return cum_cost

    def realtime_reduced_mean(self, iteration: int = 1000):
        print("<<시뮬레이션 시작>>")
        reduced_means = [self.cumulative_cost()]
        for n in range(2, iteration+1):
            if n % 500 == 0:
                print(f"iteration : {n}")
            mu = (1 - 1/n) * reduced_means[-1] + self.cumulative_cost() / n
            next_mean = int(mu)
            reduced_means.append(next_mean)
        reduced_means_nparray = np.array(reduced_means)
        self.simulated_reduced_cost = reduced_means_nparray
        return reduced_means_nparray

    def print_realtime_reduced_mean(self, iteration: int = 1000):
        print("<<시뮬레이션 시작>>")
        reduced_means = [self.cumulative_cost()]
        for n in range(2, iteration+1):
            if n % 500 == 0:
                print(f"iteration : {n}")
            mu = (1 - 1/n) * reduced_means[-1] + self.cumulative_cost() / n
            next_mean = int(mu)
            reduced_means.append(next_mean)
            print(f"{next_mean:>15,},000")
        return np.array(reduced_means)

    def draw_mean_graph(self, means: np.ndarray | None = None, guide_line: bool = False):
        if means is None:
            if self.simulated_reduced_cost is None:
                plt.plot(self.realtime_reduced_mean() / 1000 / 1000,
                         label='induced by simulation')
            else:
                plt.plot(self.simulated_reduced_cost / 1000 / 1000,
                         label='induced by simulation')
        else:
            plt.plot(means/1000/1000)

        markov_mean = self.markov_class.interval_cost["cumulative"].iloc[-1] / 1000 / 1000 / 1000
        if guide_line:
            plt.axhline(y=markov_mean, color='r', linestyle='-', label='induced by MRP')
        plt.xlabel('iteration')
        plt.ylabel('sample mean(Billion)')
        plt.ylim(plt.ylim()[0], min(plt.ylim()[1], markov_mean * 2))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # ver1 과 ver2 속도비교
    def performance_comparison():
        # version 1
        old_simulator = StarForceSimulator(0, 20, 160)
        # 1회 시뮬레이션 시간 측정
        start_time = time.time()
        old_simulator.cumulative_cost(0, 22)
        end_time = time.time()
        run_time1 = end_time - start_time
        print("version 1 :", run_time1)

        # version 2
        simulator = StarforceSimulatorVer2(0, 22, 160)
        # 1회 시뮬레이션 시간 측정
        start_time = time.time()
        simulator.cumulative_cost()
        end_time = time.time()
        run_time2 = end_time - start_time
        print("version 2 :", run_time2)

        if run_time2 == 0:
            pass
        else:
            print(f"속도 비교 : version 2가 {run_time1/run_time2} 배 빠르다.")

    performance_comparison()

    # ver2 로 그래프 그리기
    simulator = StarforceSimulatorVer2(start=0, goal=22, item_lv=160, base_price=0, starcatch=True, event30=True)
    simulator.realtime_reduced_mean(iteration=3000)
    simulator.draw_mean_graph(guide_line=True)

    # 장비파괴비용 고려하는 경우 오차의 정도
    destroy_cost_simulator = StarforceSimulatorVer2(start=0, goal=22, item_lv=160,
                                                    base_price=100000000, starcatch=False, event30=True)
    destroy_cost_simulator.realtime_reduced_mean(iteration=3000)
    destroy_cost_simulator.draw_mean_graph(guide_line=True)
    print(destroy_cost_simulator.table)
