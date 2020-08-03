from typing import List, Tuple, Callable
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from matching.eval import Eval, EvalResult
from matching.matcher import MatchModelTrainer


class SkLearnMatcher(MatchModelTrainer):
    def __init__(
        self,
        pair_to_vec: Callable[[int, int], np.ndarray],
        classifier=None,
        hint: str = None,
    ):
        self._pair_to_vec = pair_to_vec
        self._classifier = (
            classifier if classifier is not None else RandomForestClassifier()
        )
        self._eval = Eval(self._predict_pair)
        self.hint = hint

    def __repr__(self):
        return self.hint

    def __str__(self):
        return self.hint

    def fit(
        self,
        labelled_train_pairs: List[Tuple[int, int, int]],
        labelled_val_pairs: List[Tuple[int, int, int]],
    ):
        x = [self._pair_to_vec(e[0], e[1]) for e in labelled_train_pairs]
        y = [e[2] for e in labelled_train_pairs]
        self._classifier.fit(x, y)

    def predict(self, pairs: List[Tuple[int, ...]]) -> List[float]:
        return self._classifier.predict([self._pair_to_vec(e[0], e[1]) for e in pairs])

    def _predict_pair(self, e1: int, e2: int) -> float:
        return self.predict([(e1, e2)])[0]

    def evaluate(self, labelled_pairs: List[Tuple[int, int, int]]) -> EvalResult:
        prediction = self.predict(labelled_pairs)
        return self._eval.evaluate(
            labelled_pairs,
            [e[:2] for p, e in zip(prediction, labelled_pairs) if p > 0.5],
        )
