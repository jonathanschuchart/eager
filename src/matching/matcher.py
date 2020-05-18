from abc import ABC, abstractmethod
from typing import Tuple, List


class MatchModelTrainer(ABC):
    @abstractmethod
    def fit(
        self,
        labelled_train_pairs: List[Tuple[int, int, int]],
        labelled_val_pairs: List[Tuple[int, int, int]],
    ):
        pass

    @abstractmethod
    def predict(self, pairs: List[Tuple[int, int]]) -> List[float]:
        pass
