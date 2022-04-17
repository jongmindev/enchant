import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class MarkovTransitionMatrix(ABC):
    """Transition Matrix P 생성"""

    @abstractmethod
    def _make_transition_matrix(self, *args, **kwargs) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def transition_matrix(self) -> np.ndarray:
        pass

    def _transition_matrix_validation(self):
        for row in self.transition_matrix:
            if abs(sum(row) - 1.0) > 0.001 / 100:
                raise ValueError("The transition matrix is WRONG.")


class MarkovFundamentalMatrix:
    """Fundamental Matrix N = (I - Q)^(-1) 생성, where Q : sub-matrix of P"""
    def __init__(self, transition_matrix_class: MarkovTransitionMatrix):
        self._transition_matrix = transition_matrix_class.transition_matrix.copy()
        self._sub_matrix_q = self._transition_matrix[:-1, :-1].copy()
        self._fundamental_matrix = self._make_fundamental_matrix()

    def _make_fundamental_matrix(self) -> np.ndarray:
        identity = np.identity(len(self._sub_matrix_q), dtype=float)
        fundamental_matrix = np.linalg.inv(identity - self._sub_matrix_q)
        return fundamental_matrix

    @property
    def fundamental_matrix(self) -> np.ndarray:
        return self._fundamental_matrix


class MarkovReward:
    """expected total reward before absorption"""
    def __init__(self, transition_matrix_class: MarkovTransitionMatrix, reward: np.ndarray, col_name: str):
        self._fundamental_matrix = MarkovFundamentalMatrix(transition_matrix_class).fundamental_matrix.copy()
        differential_expected_reward_before_absorption = self._expected_reward_array_for_next_step(reward)
        self._TOTAL_REWARD = pd.DataFrame(data=differential_expected_reward_before_absorption, columns=[col_name])
        self._TOTAL_REWARD.index.name = 'from'

    def _expected_reward_array_for_next_step(self, reward: np.ndarray) -> np.ndarray:
        cumulative_expected_reward = self._fundamental_matrix @ reward
        differential_expected_reward_before_absorption = - np.diff(a=cumulative_expected_reward, append=0)
        return differential_expected_reward_before_absorption

    @property
    def TOTAL_REWARD(self) -> pd.DataFrame:
        return self._TOTAL_REWARD


class MarkovMean:
    """expected total (try) times before absorption : special class of MarkovReward
    Because of using MarkovReward class, this is not optimized considering round-off error.
    Summation of row of N is better than calculating N @ 1."""
    def __init__(self, transition_matrix_class: MarkovTransitionMatrix):
        _fundamental_matrix = MarkovFundamentalMatrix(transition_matrix_class).fundamental_matrix
        reward = np.ones(len(_fundamental_matrix[0]), dtype=float)
        self._MEAN_DF = MarkovReward(transition_matrix_class, reward, "expected times").TOTAL_REWARD

    @property
    def MEAN_DF(self) -> pd.DataFrame:
        return self._MEAN_DF


# class MarkovStatistic:
#     """Markov Chain 평균, (분산) 계산"""
#     def __init__(self, transition_matrix_class: MarkovTransitionMatrix):
#         self._transition_matrix = transition_matrix_class.transition_matrix.copy()
#         self._fundamental_matrix = MarkovFundamentalMatrix(transition_matrix_class).fundamental_matrix
#         self._STATISTICS = self._make_statistics_df()
#
#     def _make_expected_time_list(self) -> np.ndarray:
#         means_to_absorbing_stage = np.array([sum(row) for row in self._fundamental_matrix])
#         temp = np.append(arr=means_to_absorbing_stage[1:], values=0.)
#         return means_to_absorbing_stage - temp
#
#     def _make_statistics_df(self) -> pd.DataFrame:
#         df = pd.DataFrame()
#         df.index.name = "from"
#         df["expected time"] = self._make_expected_time_list()
#         return df
#
#     @property
#     def STATISTICS(self):
#         return self._STATISTICS
