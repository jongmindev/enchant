import markov
import table
import numpy as np


class StarForceTransitionMatrix(markov.MarkovTransitionMatrix):
    def __init__(self, absorbing_stage, starcatch=False, prevent1216=(False, False, False, False, False),
                 event51015=False, event1plus1: bool=False):
        self._transition_matrix = self._make_transition_matrix(absorbing_stage,
                                                               starcatch, prevent1216, event51015, event1plus1)
        self._transition_matrix_validation()

    def _make_transition_matrix(self, absorbing_stage, starcatch, prevent1216, event51015, event1plus1) -> np.ndarray:
        prob_table_df = table.StarForceTable(starcatch, prevent1216, event51015).prob_table

        up_prob = np.array(prob_table_df["up"])[:absorbing_stage]
        keep_prob = np.array(prob_table_df["keep"])[:absorbing_stage]
        down_prob = np.array(prob_table_df["down"])[:absorbing_stage]
        destroy_prob = np.array(prob_table_df["destroy"])[:absorbing_stage]

        upper_diagonal = up_prob[:]
        main_diagonal = np.append(keep_prob, 1.)
        lower_diagonal = np.append(down_prob[1:], 0.)

        transition_matrix = np.diag(upper_diagonal, k=1) + np.diag(main_diagonal) + np.diag(lower_diagonal, k=-1)

        # 파괴확률 반영
        destroy_column = np.append(destroy_prob, 0.)
        destroy_matrix = np.zeros(transition_matrix.shape)
        destroy_matrix[:, 12] = destroy_column
        transition_matrix += destroy_matrix

        # 10성 이하 1+1 이벤트 반영
        if event1plus1:
            for i in range(11):
                transition_matrix[i, i+1], transition_matrix[i, i+2] = \
                    transition_matrix[i, i+2], transition_matrix[i, i+1]

        return transition_matrix

    @property
    def transition_matrix(self) -> np.ndarray:
        return self._transition_matrix


if __name__ == "__main__":
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    GOAL = 25

    starforce_transition_matrix_class = StarForceTransitionMatrix(absorbing_stage=GOAL)
    P = starforce_transition_matrix_class.transition_matrix
    print(pd.DataFrame(P)*100)

    starforce_transition_matrix_class2 = StarForceTransitionMatrix(absorbing_stage=GOAL, event1plus1=True)
    P2 = starforce_transition_matrix_class2.transition_matrix
    print(pd.DataFrame(P2)*100)
