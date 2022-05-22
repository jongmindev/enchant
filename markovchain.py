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

    # transition matrix 가 잘 만들어졌는지 확인하는 method : 각 행의 합이 1이 맞는지
    def _transition_matrix_validation(self):
        for row in self.transition_matrix:
            if abs(sum(row) - 1.0) > 0.001 / 100:
                raise ValueError("The transition matrix is WRONG.")

    def print_transition_matrix(self):
        pd.options.display.max_columns = None
        pd.options.display.expand_frame_repr = False
        pd.options.display.float_format = '{:.2f}'.format
        df = pd.DataFrame(self.transition_matrix).replace(0., "")
        print(df * 100)
        pd.reset_option("display.max_columns")
        pd.reset_option("display.expand_frame_repr")
        pd.reset_option("display.float_format")


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
        self._total_reward_df = pd.DataFrame(data=differential_expected_reward_before_absorption, columns=[col_name])
        self._total_reward_df.index.name = 'from'

    def _expected_reward_array_for_next_step(self, reward: np.ndarray) -> np.ndarray:
        cumulative_expected_reward = self._fundamental_matrix @ reward
        differential_expected_reward_before_absorption = - np.diff(a=cumulative_expected_reward, append=0)
        return differential_expected_reward_before_absorption

    @property
    def total_reward_df(self) -> pd.DataFrame:
        return self._total_reward_df


class MarkovMean:
    """expected total (try) times before absorption : special class of MarkovReward
    Because of using MarkovReward class, this is not optimized considering round-off error.
    Summation of row of N is better than calculating N @ 1."""
    def __init__(self, transition_matrix_class: MarkovTransitionMatrix):
        _fundamental_matrix = MarkovFundamentalMatrix(transition_matrix_class).fundamental_matrix
        reward = np.ones(len(_fundamental_matrix[0]), dtype=float)
        self._mean_df = MarkovReward(transition_matrix_class, reward, "expected times").total_reward_df

    @property
    def mean_df(self) -> pd.DataFrame:
        return self._mean_df
