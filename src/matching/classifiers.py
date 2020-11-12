import os
import pickle
from typing import List, Tuple, Callable
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from matching.eval import Eval, EvalResult
from matching.matcher import MatchModelTrainer
from matching.pair_to_vec import PairToVec


class SkLearnMatcher(MatchModelTrainer):
    def __init__(
        self, classifier=None, hint: str = None,
    ):
        # self.pair_to_vec = pair_to_vec
        self._classifier = (
            classifier if classifier is not None else RandomForestClassifier()
        )
        self._eval = Eval(self._predict_pair)
        self.hint = hint

    def __repr__(self):
        return f"{self.hint} - {self.pair_to_vec.name}"

    def __str__(self):
        return f"{self.hint} - {self.pair_to_vec.name}"

    def fit(
        self,
        labelled_train_pairs: List[Tuple[int, int, int]],
        labelled_val_pairs: List[Tuple[int, int, int]],
    ):
        x = [self.pair_to_vec(e[0], e[1]) for e in labelled_train_pairs]
        y = [e[2] for e in labelled_train_pairs]
        self._classifier.fit(x, y)

    def predict(self, pairs: List[Tuple[int, ...]]) -> List[float]:
        return self._classifier.predict([self.pair_to_vec(e[0], e[1]) for e in pairs])

    def _predict_pair(self, e1: int, e2: int) -> float:
        return self.predict([(e1, e2)])[0]

    def evaluate(self, labelled_pairs: List[Tuple[int, int, int]]) -> EvalResult:
        prediction = self.predict(labelled_pairs)
        return self._eval.evaluate(
            labelled_pairs,
            [(e[0], e[1], e[2], p) for p, e in zip(prediction, labelled_pairs)],
        )

    # TODO: finish save/load
    def save(self, folder):
        with open(os.path.join(folder, self.hint), "w") as f:
            pickle.dump(self._classifier, f)
        with open(os.path.join(folder, type(self.pair_to_vec).__name__), "w") as f:
            pickle.dump(self.pair_to_vec, f)
        # with open()

    @staticmethod
    def load(folder):
        with open(os.path.join(folder, "classifier.pkl")) as classifier:
            with open(os.path.join(folder, "pvp_name")) as pvp:
                return SkLearnMatcher(pickle.load(pvp), pickle.load(classifier))
