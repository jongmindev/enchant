import markov
import table
import numpy as np


class GameMangTransitionMatrix(markov.MarkovTransitionMatrix):
    def __init__(self, absorbing_stage, water=False):
        self._transition_matrix = self._make_transition_matrix(absorbing_stage, water)
        self._transition_matrix_validation()

    def _make_transition_matrix(self, absorbing_stage, water) -> np.ndarray:
        prob_table_df = table.GameMangTable(water).prob_table
        up_prob = np.array(prob_table_df["up"])[:absorbing_stage]
        keep_prob = np.array(prob_table_df["keep"])[:absorbing_stage]
        down_prob = np.array(prob_table_df["down"])[:absorbing_stage]

        upper_diagonal = up_prob
        main_diagonal = np.append(keep_prob, 1.)
        lower_diagonal = np.append(down_prob[1:], 0.)

        transition_matrix = np.diag(upper_diagonal, k=1) + np.diag(main_diagonal) + np.diag(lower_diagonal, k=-1)

        return transition_matrix

    @property
    def transition_matrix(self) -> np.ndarray:
        return self._transition_matrix


if __name__ == "__main__":
    GOAL = 13
    gamemang_transition_matrix_class = GameMangTransitionMatrix(absorbing_stage=GOAL, water=False)
    gamemang_statistics = markov.MarkovStatistic(gamemang_transition_matrix_class)
    no_water_df = gamemang_statistics.STATISTICS
    print("NO WATER: \n", no_water_df)
    print()

    gamemang_transition_matrix_class2 = GameMangTransitionMatrix(absorbing_stage=GOAL, water=True)
    gamemang_statistics2 = markov.MarkovStatistic(gamemang_transition_matrix_class2)
    yes_water_df = gamemang_statistics2.STATISTICS
    print("YES WATER: \n", yes_water_df)
    print()

    print(table.GameMangTable(water=True).prob_table)
    print()

    no_water_is_cheap = no_water_df - 10 * yes_water_df
    print(no_water_is_cheap)
