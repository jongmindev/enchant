import markov
import gamemang
import starforce
import pandas as pd


# GameMang enchant option : goal, water
class GameMangCalculator:
    """mandatory parameter(1) : goal    /    optional parameter(1) : water"""
    def __init__(self, goal: int, water: bool | None = None):
        if water == None:
            transition_no = gamemang.GameMangTransitionMatrix(absorbing_stage=goal, water=False)
            transition_yes = gamemang.GameMangTransitionMatrix(absorbing_stage=goal, water=True)
            cost_no = markov.MarkovMean(transition_no).mean_df
            cost_no.rename(columns={cost_no.columns[0]: "no water"}, inplace=True)
            cost_yes = markov.MarkovMean(transition_yes).mean_df
            cost_yes.rename(columns={cost_yes.columns[0]: "yes water"}, inplace=True)
            cost_yes.iloc[3:, -1] *= 10
            self._mean = pd.merge(cost_no, cost_yes, left_index=True, right_index=True)
            self._mean["no is better"] = self._mean["no water"] <= self._mean["yes water"]
        else:
            transition = gamemang.GameMangTransitionMatrix(absorbing_stage=goal, water=water)
            self._mean = markov.MarkovMean(transition).mean_df.copy()
            if water:
                self._mean.rename(columns={self._mean.columns[0]: "yes water"}, inplace=True)
                self._mean.iloc[3:, -1] *= 10
            else:
                self._mean.rename(columns={self._mean.columns[0]: "no water"}, inplace=True)

    @property
    def mean(self):
        return self._mean


# StarForce option :
# goal, item_lv, base_price, starcatch, mvp, pc_room, event30, event51015, event1plus1, prevent1216
class StarForceCalculator:
    """mandatory parameters(2) : goal, item_lv
    optional parameters(8) : base_price, starcatch, mvp, pc_room, event30, event51015, event1plus1, prevent1216"""
    def __init__(self, goal: int, item_lv: int, base_price: int = 0,
                 starcatch: bool = False, mvp: str = "bronze", pc_room: bool = False,
                 event30: bool = False, event51015: bool = False, event1plus1: bool = False,
                 prevent1216: tuple = (False, False, False, False, False)):
        transition = starforce.StarForceTransitionMatrix(goal, starcatch, prevent1216, event51015, event1plus1)
        cost_per_1 = starforce.StarForceCost(item_lv, mvp, pc_room, event30, prevent1216).reward[:goal]
        total_cost_protype = markov.MarkovReward(transition, cost_per_1, "Expected Interval Cost").total_reward_df
        self._total_cost_prototype = total_cost_protype
        # 기대 파괴횟수 : 미구현
        expected_destroy_times = 0
        self._total_cost = total_cost_protype + base_price * expected_destroy_times

    @property
    def total_cost(self):
        return self._total_cost


if __name__ == "__main__":
    # help(GameMangCalculator)
    # help(StarForceCalculator)

    pd.options.display.float_format = '{:.2f}'.format
    print(GameMangCalculator(goal=20).mean)
    print(GameMangCalculator(goal=20, water=True).mean)
    print(GameMangCalculator(goal=20, water=False).mean)
    print(StarForceCalculator(goal=22, item_lv=160).total_cost)
