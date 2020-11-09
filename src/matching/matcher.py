from abc import ABC, abstractmethod
from typing import Tuple, List

from matching.eval import EvalResult


class MatchModelTrainer(ABC):
    @abstractmethod
    def fit(
        self,
        labelled_train_pairs: List[Tuple[int, int, int]],
        labelled_val_pairs: List[Tuple[int, int, int]],
    ):
        pass

    @abstractmethod
    def predict(self, pairs: List[Tuple]) -> List[float]:
        pass

    @abstractmethod
    def evaluate(self, labelled_pairs: List[Tuple[int, int, int]]) -> EvalResult:
        pass

    @abstractmethod
    def save(self, folder):
        pass

    @staticmethod
    @abstractmethod
    def load(folder):
        pass
