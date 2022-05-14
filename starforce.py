import markovchain
import table
import numpy as np
import pandas as pd


class StarForceTransitionMatrix(markov.MarkovTransitionMatrix):
    def __init__(self, absorbing_stage, starcatch=False, prevent1216=(False, False, False, False, False),
                 event51015=False, event1plus1: bool = False):
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


class StarForceCost:
    def __init__(self, item_lv: int, mvp: str = "bronze", pc_room: bool = False,
                 event30: bool = False, prevent1216=(False, False, False, False, False)):
        self._reward = self._make_reward_array(item_lv, mvp, pc_room, event30, prevent1216)

    @staticmethod
    def _make_reward_array(item_lv: int, mvp: str, pc_room: bool,
                           event30: bool, prevent1216: tuple) -> np.ndarray:
        # 기본비용
        rewards = np.zeros(25, dtype=int)
        for i in range(0, 10):
            rewards[i] = int(round(1000 + item_lv**3 * (i + 1) / 25, ndigits=-2))
        for i in range(10, 15):
            rewards[i] = int(round(1000 + item_lv**3 * (i + 1)**2.7 / 400, ndigits=-2))
        for i in range(15, 25):
            rewards[i] = int(round(1000 + item_lv**3 * (i + 1)**2.7 / 200, ndigits=-2))

        modified = rewards.copy()

        def ratio_discount(_rewards: np.ndarray, ratio: float):
            discounted = _rewards * ratio
            discounted = discounted.round(-2).astype(int)
            return discounted

        # mvp 할인과 pc방 할인은 합 적용
        if mvp == "bronze":
            if pc_room:
                modified[:17] = ratio_discount(modified[:17], 0.95)
            else:
                pass
        elif mvp == "silver":
            if pc_room:
                modified[:17] = ratio_discount(modified[:17], 0.92)
            else:
                modified[:17] = ratio_discount(modified[:17], 0.97)
        elif mvp == "gold":
            if pc_room:
                modified[:17] = ratio_discount(modified[:17], 0.9)
            else:
                modified[:17] = ratio_discount(modified[:17], 0.95)
        elif (mvp == "diamond") | (mvp == "red"):
            if pc_room:
                modified[:17] = ratio_discount(modified[:17], 0.85)
            else:
                modified[:17] = ratio_discount(modified[:17], 0.9)
        else:
            raise ValueError("Parameter 'mvp' should be one of ['bronze', 'silver', 'gold', 'diamond', 'red'].")

        # 전구간 30% 할인
        if event30:
            modified = ratio_discount(modified, 0.7)
        else:
            pass

        # 파괴방지 : 할인된 금액에 기본비용의 100%를 더한 값
        for i in range(len(prevent1216)):
            if prevent1216[i]:
                star = i + 12
                modified[star] += rewards[star]

        return modified

    @property
    def reward(self):
        return self._reward

    def reward_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._reward).rename(columns={0: "interval_cost"})


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

    cost150 = StarForceCost(item_lv=150, mvp="gold", event30=True).reward
    df = pd.DataFrame(cost150, columns=["cost"])
    df.index.name = "from"
    print(df)
