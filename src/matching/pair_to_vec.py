from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import numpy as np
from openea.modules.load.kgs import KGs

from similarity.similarities import calculate_on_demand


class PairToVec(ABC):
    def __init__(
        self,
        embeddings: np.ndarray,
        all_sims: Dict[Tuple[int, int], Dict[str, float]],
        kgs: KGs,
    ):
        self.embeddings = embeddings
        self.all_sims = all_sims
        self.all_keys = {k for v in all_sims.values() for k in v.keys()}
        self.kgs = kgs

    @abstractmethod
    def __call__(self, e1: int, e2: int) -> np.ndarray:
        pass

    @abstractmethod
    def dimension(self) -> int:
        pass


class SimAndEmb(PairToVec):
    def __call__(self, e1: int, e2: int) -> np.ndarray:
        sim = self.all_sims.get((e1, e2))
        if sim is None:
            sim = calculate_on_demand(self.embeddings, (e1, e2), self.kgs, "euclidean")
        sim_vec = np.asarray([sim.get(k, 0) for k in self.all_keys])
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
        sim_vec = np.asarray([sim.get(k, 0) for k in self.all_keys])
        return sim_vec

    def dimension(self) -> int:
        return len(self.all_keys)


class OnlySimNormalized(PairToVec):
    def __call__(self, e1: int, e2: int) -> np.ndarray:
        sim = self.all_sims.get((e1, e2))
        if sim is None:
            sim = calculate_on_demand(self.embeddings, (e1, e2), self.kgs, "euclidean")
        sim_vec = np.asarray([sim.get(k, 0) for k in self.all_keys])
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
