import markovchain
import gamemang
import starforce
import pandas as pd
from abc import ABC, abstractmethod


class Calculator(ABC):
    @property
    @abstractmethod
    def interval_cost(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def print_interval_cost(self):
        pass


# GameMang enchant option : goal, water
class GameMangCalculator(Calculator):
    """mandatory parameter(1) : goal    /    optional parameter(1) : water"""
    def __init__(self, goal: int, water: bool | None = None):
        if water is None:
            transition_no = gamemang.GameMangTransitionMatrix(absorbing_stage=goal, water=False)
            transition_yes = gamemang.GameMangTransitionMatrix(absorbing_stage=goal, water=True)
            cost_no = markovchain.MarkovMean(transition_no).mean_df
            cost_no.rename(columns={cost_no.columns[0]: "no water"}, inplace=True)
            cost_yes = markovchain.MarkovMean(transition_yes).mean_df
            cost_yes.rename(columns={cost_yes.columns[0]: "yes water"}, inplace=True)
            cost_yes.iloc[3:, -1] *= 10
            merged = pd.merge(cost_no, cost_yes, left_index=True, right_index=True)
            merged.loc[3:, "no is better"] = merged["no water"] < merged["yes water"]
            mask = merged["no is better"] == True
            merged.loc[mask, "profit"] = merged["yes water"] - merged["no water"]
            self._interval_cost = merged
        else:
            transition = gamemang.GameMangTransitionMatrix(absorbing_stage=goal, water=water)
            self._interval_cost = markovchain.MarkovMean(transition).mean_df.copy()
            if water:
                self._interval_cost.rename(columns={self._interval_cost.columns[0]: "yes water"}, inplace=True)
                self._interval_cost.iloc[3:, -1] *= 10
                self._interval_cost["cumulative"] = self._interval_cost["yes water"].cumsum()
            else:
                self._interval_cost.rename(columns={self._interval_cost.columns[0]: "no water"}, inplace=True)
                self._interval_cost["cumulative"] = self._interval_cost["no water"].cumsum()

    @property
    def interval_cost(self):
        return self._interval_cost

    def print_interval_cost(self):
        pd.options.display.float_format = '{:,.2f}'.format
        print(self._interval_cost)
        pd.reset_option("display.float_format")


# StarForce option :
# goal, item_lv, base_price, starcatch, mvp, pc_room, event30, event51015, event1plus1, prevent1216
class StarForceCalculator(Calculator):
    """mandatory parameters(2) : goal, item_lv
    optional parameters(8) : base_price, starcatch, mvp, pc_room, event30, event51015, event1plus1, prevent1216"""
    def __init__(self, goal: int, item_lv: int, base_price: int = 0,
                 starcatch: bool = False, mvp: str = "bronze", pc_room: bool = False,
                 event30: bool = False, event51015: bool = False, event1plus1: bool = False,
                 prevent1216: tuple = (False, False, False, False, False)):
        transition = starforce.StarForceTransitionMatrix(goal, starcatch, prevent1216, event51015, event1plus1)
        cost_per_1 = starforce.StarForceCost(item_lv, mvp, pc_room, event30, prevent1216).reward[:goal]
        interval_cost_protype = markovchain.MarkovReward(transition, cost_per_1, "Expected Interval Cost").total_reward_df
        self._interval_cost_prototype = interval_cost_protype
        # 기대 파괴횟수 : 미구현
        expected_destroy_times = 0
        self._interval_cost = interval_cost_protype + base_price * expected_destroy_times
        self._interval_cost["cumulative"] = self._interval_cost["Expected Interval Cost"].cumsum()

    @property
    def interval_cost(self):
        return self._interval_cost

    def print_interval_cost(self):
        pd.options.display.float_format = '{:,.0f}'.format
        print(self._interval_cost)
        pd.reset_option("display.float_format")


if __name__ == "__main__":
    GameMangCalculator(goal=20).print_interval_cost()
    GameMangCalculator(goal=20, water=True).print_interval_cost()
    GameMangCalculator(goal=20, water=False).print_interval_cost()
    StarForceCalculator(goal=22, item_lv=160).print_interval_cost()
    StarForceCalculator(goal=13, item_lv=160).print_interval_cost()
    # cost22 = StarForceCalculator(goal=22, item_lv=160).interval_cost[:13]
    # cost13 = StarForceCalculator(goal=13, item_lv=160).interval_cost
    # print(abs(cost13 - cost22))
    # starforce.StarForceTransitionMatrix(absorbing_stage=22).print_transition_matrix()
    # starforce.StarForceTransitionMatrix(absorbing_stage=13).print_transition_matrix()
    # print(pd.DataFrame(starforce.StarForceCost(160).reward))
    # StarForceCalculator(goal=22, item_lv=160, prevent1216=(True, True, True, True, True,)).print_interval_cost()
    tester = StarForceCalculator(goal=22, item_lv=160)
    print(tester.interval_cost.cumulative)

    StarForceCalculator(goal=22, item_lv=160, base_price=0,
                        starcatch=True, mvp="bronze", pc_room=False,
                        event30=True, event51015=False, event1plus1=False,
                        prevent1216=(False, False, False, False, False)).print_interval_cost()

    StarForceCalculator(goal=22, item_lv=160, base_price=0,
                        starcatch=True, mvp="bronze", pc_room=False,
                        event30=False, event51015=False, event1plus1=False,
                        prevent1216=(False, False, False, False, False)).print_interval_cost()
