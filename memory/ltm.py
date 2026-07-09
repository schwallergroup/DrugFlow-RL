from typing import Tuple, List
import numpy as np
import pandas as pd
from torch_geometric.data import Data


class LongTimeMem:
    """
    Long-term memory class which stores all generated SMILES so far.
        1. SMILES strings
        2. Reward
        3. RL step
    """

    def __init__(
        self,
        # df_tol: int = 10,
        # df_buff: int = 20,
    ) -> None:
        # self.df_tol = df_tol
        # self.df_buff = df_buff
        # assert self.df_tol < self.df_buff

        # Stores all generated SMILES and rewards
        self.memory = pd.DataFrame(
            columns=["smiles", "reward", "RL_step"]
        )
        self.unique_smiles = []

    def extend(self, smiles: np.ndarray[str], rewards: np.ndarray[float], step: int) -> None:
        """
        Extend memory with an array of SMILES strings and their corresponding rewards.
        
        Args:
            smiles: Array of SMILES strings
            rewards: Array of reward values corresponding to each SMILES
            step: Current RL step
        """
        steps = [step for _ in smiles]
        
        df_sample = pd.DataFrame.from_dict({
            "smiles": smiles,
            "reward": rewards,
            "RL_step": steps,
        })
        if len(self.memory) > 0:
            self.memory = pd.concat([self.memory, df_sample])
        else:
            self.memory = df_sample

        # save unique smiles
        self.unique_smiles = self.memory["smiles"].unique().tolist()

    def div_filter(
        self,
        smiles: np.ndarray[str],
        rewards: np.ndarray[float],
        tol: int = 10,
        buff: int = 20,
        **kwargs,
    ) -> Tuple[np.ndarray[float], List[int], int, int]:
        """
        Apply diversity filtering to SMILES strings based on occurrence frequency.
        
        Args:
            smiles: Array of SMILES strings
            rewards: Array of reward values
            tol: Tolerance threshold
            buff: Buffer threshold
        """
        # ref: Augmented Hill-Climb, https://doi.org/10.1186/s13321-022-00646-z
        # tol = tolerance, buff = buffer, occ = occurrences
        assert tol < buff

        new_rewards = []
        penalty_idx = []
        tol_n = 0
        buff_n = 0
        for i, smile in enumerate(smiles):
            occ = self.memory["smiles"].value_counts().get(smile, 0)
            if occ <= tol:
                new_rewards.append(rewards[i])
            elif occ > tol and occ < buff:
                new_rewards.append(
                    rewards[i] * (buff - occ) / (buff - tol)
                )
                tol_n += 1
            else:
                new_rewards.append(0.0)
                penalty_idx.append(i)
                buff_n += 1

        return np.array(new_rewards), penalty_idx, tol_n, buff_n

    def calc_metrics(
        self,
        thred: float,
        budget: int = 3000,
        num_candidate: int = 100,
    ) -> Tuple[float, float]:
        # Burden metric
        _df = self.memory.sort_values("reward", ascending=False)
        unique_df = _df.drop_duplicates(subset=["smiles"])
        candidates = (unique_df["reward"] > thred).sum()
        calc_cost = len(self.memory)
        if candidates >= num_candidate:
            burden = calc_cost / candidates
        else:
            burden = None

        # Diversity ratio
        num_unique_smiles = len(self.unique_smiles)
        if calc_cost <= budget:
            div_ratio = num_unique_smiles / calc_cost
        else:
            div_ratio = None

        return burden, div_ratio

    def get_baseline(self, step: int, prev: int = 3):
        baseline = self.memory[self.memory["RL_step"] > step - prev]["reward"].mean()
        return baseline

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate SMILES strings.
        Keep only non-zero rewards SMILES.
        """
        _df = df.sort_values("reward", ascending=False)
        unique_df = _df.drop_duplicates(subset=["smiles"])
        return unique_df

    def save(self, save_path: str):
        self.memory.to_csv(save_path, index=False)

    def __len__(self) -> int:
        return len(self.memory)
