import pandas as pd
from abc import ABC, abstractmethod


class ProbabilityTable(ABC):
    @property
    @abstractmethod
    def prob_table(self) -> pd.DataFrame:
        pass

    def table_validation(self):
        for iterrow in self.prob_table.iterrows():
            row = iterrow[1]
            if abs(row.sum() - 1) > 0.001 / 100:
                raise ValueError("The probability table is WRONG.")


class GameMangTable(ProbabilityTable):
    standard_table = pd.read_csv("data/gamemang_elementary_table.csv", index_col=["from"])

    def __init__(self, water=False):
        self._prob_table = GameMangTable.set_modified_table(water)
        self.table_validation()

    @property
    def prob_table(self) -> pd.DataFrame:
        return self._prob_table

    @classmethod
    def set_modified_table(cls, water=False):
        modified = GameMangTable.standard_table.copy()
        if water:
            modified["keep"] = modified["down"]
            modified["down"] = 0
        else:
            pass
        return modified


class StarForceTable(ProbabilityTable):
    standard_table = pd.read_csv("data/starforce_elementary_table.csv", index_col=["from"])

    def __init__(self, starcatch=False, prevent1216=(False, False, False, False, False), event51015=False):
        self._prob_table = StarForceTable.set_modified_table(starcatch, prevent1216, event51015)
        self.table_validation()

    @property
    def prob_table(self) -> pd.DataFrame:
        return self._prob_table

    @classmethod
    def up_probs(cls, starcatch=False):
        """성공확률 계산"""
        modified = StarForceTable.standard_table["up"].copy()
        # 스타캐치
        if starcatch:
            modified *= 1.05
        else:
            pass
        return modified

    @classmethod
    def destroy_probs(cls, starcatch=False, prevent1216=(False, False, False, False, False)):
        """파괴확률 계산"""
        modified = StarForceTable.standard_table["destroy"].copy()
        up_modified = StarForceTable.up_probs(starcatch)
        up_standard = StarForceTable.standard_table["up"].copy()
        # 스타캐치
        if starcatch:
            modified *= (1 - up_modified) / (1 - up_standard)
        else:
            pass
        # 파괴방지
        modified.loc[12:16][list(prevent1216)] = 0.0
        return modified

    @classmethod
    def keep_probs(cls, starcatch=False, prevent1216=(False, False, False, False, False)):
        """유지확률 계산"""
        modified = StarForceTable.standard_table["keep"].copy()
        up_modified = StarForceTable.up_probs(starcatch)
        destroy_modified = StarForceTable.destroy_probs(starcatch, prevent1216)
        modified.loc[0:10] = 1 - up_modified.loc[0:10]
        modified.loc[15] = 1 - up_modified.loc[15] - destroy_modified.loc[15]
        modified.loc[20] = 1 - up_modified.loc[20] - destroy_modified.loc[20]
        return modified

    @classmethod
    def down_probs(cls, starcatch=False, prevent1216=(False, False, False, False, False)):
        """하락확률 계산"""
        modified = StarForceTable.standard_table["down"].copy()
        up_modified = StarForceTable.up_probs(starcatch)
        destroy_modified = StarForceTable.destroy_probs(starcatch, prevent1216)
        modified.loc[11:14] = 1 - up_modified.loc[11:14] - destroy_modified.loc[11:14]
        modified.loc[16:19] = 1 - up_modified.loc[16:19] - destroy_modified.loc[16:19]
        modified.loc[21:24] = 1 - up_modified.loc[21:24] - destroy_modified.loc[21:24]
        return modified

    @classmethod
    def set_modified_table(cls, starcatch=False, prevent1216=(False, False, False, False, False), event51015=False):
        total_modified = StarForceTable.standard_table.copy()
        total_modified["up"] = StarForceTable.up_probs(starcatch)
        total_modified["keep"] = StarForceTable.keep_probs(starcatch, prevent1216)
        total_modified["down"] = StarForceTable.down_probs(starcatch, prevent1216)
        total_modified["destroy"] = StarForceTable.destroy_probs(starcatch, prevent1216)
        if event51015:
            total_modified.loc[5] = (1., 0., 0., 0.)
            total_modified.loc[10] = (1., 0., 0., 0.)
            total_modified.loc[15] = (1., 0., 0., 0.)
        else:
            pass
        return total_modified


if __name__ == "__main__":
    print("STANDARD")
    print(StarForceTable.standard_table)
    print()

    print("STARCATCH")
    print(StarForceTable(starcatch=True).prob_table)
    print()

    print("EVENT51015")
    print(StarForceTable(event51015=True).prob_table)
    print()

    print("STARCATCH & EVENT51015")
    print(StarForceTable(starcatch=True, event51015=True).prob_table)
    print()

    print("PREVENT")
    print(StarForceTable(prevent1216=(True, True, True, True, True, )).prob_table)
    print()

    print("GAMEMANG NO WATER")
    print(GameMangTable(water=False).prob_table)
    print()

    print("GAMEMANG WATER")
    print(GameMangTable(water=True).prob_table)
    print()
