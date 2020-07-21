from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from openea.modules.load.kgs import KGs
from sklearn.preprocessing import MinMaxScaler

from similarity.create_training import create_similarity_frame_on_demand
from similarity.similarities import calculate_on_demand


class PairToVec(ABC):
    def __init__(
        self,
        embeddings: np.ndarray,
        all_sims: pd.DataFrame,
        min_max: MinMaxScaler,
        scale_cols: List[str],
        kgs: KGs,
    ):
        self.embeddings = embeddings
        self.all_sims = all_sims
        self.all_keys = [c for c in self.all_sims.columns if c != "label"]
        self.min_max = min_max
        self.cols = scale_cols
        self.kgs = kgs

    @abstractmethod
    def __call__(self, e1: int, e2: int) -> np.ndarray:
        pass

    @abstractmethod
    def dimension(self) -> int:
        pass


class SimAndEmb(PairToVec):
    def __call__(self, e1: int, e2: int) -> np.ndarray:
        if (e1, e2) in self.all_sims.index:
            sim = self.all_sims.loc[(e1, e2)]
        else:
            sim = create_similarity_frame_on_demand(
                self.embeddings, (e1, e2, 0), self.kgs, self.min_max, self.cols
            )
        sim_vec = np.asarray(sim[self.all_keys].fillna(0))
        return np.concatenate(
            [sim_vec, self.embeddings[int(e1)], self.embeddings[int(e2)]]
        )

    def dimension(self) -> int:
        return len(self.all_keys) + 2 * self.embeddings.shape[1]


class SimAndEmbNormalized(PairToVec):
    def __call__(self, e1: int, e2: int) -> np.ndarray:
        sim = self.all_sims.get((e1, e2))
        if sim is None:
            sim = calculate_on_demand(self.embeddings, (e1, e2), self.kgs, "euclidean")
        sim_vec = np.asarray([sim.get(k, 0) for k in self.all_keys])
        norm = np.linalg.norm(sim_vec)
        sim_vec = sim_vec / (norm if norm > 0 else 1)
        return np.concatenate(
            [sim_vec, self.embeddings[int(e1)], self.embeddings[int(e2)]]
        )

    def dimension(self) -> int:
        return len(self.all_keys) + 2 * self.embeddings.shape[1]


class OnlySim(PairToVec):
    def __call__(self, e1: int, e2: int) -> np.ndarray:
        sim = self.all_sims.get((e1, e2))
        if sim is None:
            sim = calculate_on_demand(self.embeddings, (e1, e2), self.kgs, "euclidean")
        sim_vec = np.asarray([sim.get(k, 0) for k in self.all_keys if k != "euclidean"])
        return sim_vec

    def dimension(self) -> int:
        return len(self.all_keys)


class OnlySimNormalized(PairToVec):
    def __call__(self, e1: int, e2: int) -> np.ndarray:
        sim = self.all_sims.get((e1, e2))
        if sim is None:
            sim = calculate_on_demand(self.embeddings, (e1, e2), self.kgs, "euclidean")
        sim_vec = np.asarray([sim.get(k, 0) for k in self.all_keys if k != "euclidean"])
        norm = np.linalg.norm(sim_vec)
        sim_vec = sim_vec / (norm if norm > 0 else 1)
        return sim_vec

    def dimension(self) -> int:
        return len(self.all_keys)


class OnlyEmb(PairToVec):
    def __call__(self, e1: int, e2: int) -> np.ndarray:
        return np.concatenate([self.embeddings[int(e1)], self.embeddings[int(e2)]])

    def dimension(self) -> int:
        return 2 * self.embeddings.shape[1]
