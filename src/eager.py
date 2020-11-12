import os
from typing import List, Tuple

from matching.eval import EvalResult, Eval
from matching.pair_to_vec import PairToVec


class Eager:
    def __init__(self, classifier, pair_to_vec: PairToVec, hint: str):
        self._classifier = classifier
        self.pair_to_vec = pair_to_vec
        self._hint = hint
        self._eval = Eval(self._predict_pair)

    def __repr__(self):
        return f"{self._hint} - {self.pair_to_vec.name}"

    def __str__(self):
        return f"{self._hint} - {self.pair_to_vec.name}"

    def fit(self, train_pairs, val_pairs):
        x = [self.pair_to_vec(e[0], e[1]) for e in train_pairs]
        y = [e[2] for e in train_pairs]
        self._classifier.fit(x, y)

    def predict(self, pairs):
        return [self._predict_pair(e[0], e[1]) for e in pairs]

    def _predict_pair(self, e1: int, e2: int) -> float:
        return self._classifier.predict([self.pair_to_vec(e1, e2)])[0]

    def evaluate(self, labelled_pairs: List[Tuple[int, int, int]]) -> EvalResult:
        prediction = self.predict(labelled_pairs)
        return self._eval.evaluate(
            labelled_pairs,
            [(e[0], e[1], e[2], p) for p, e in zip(prediction, labelled_pairs)],
        )

    @staticmethod
    def _create_path(folder):
        if not os.path.exists(folder):
            one_up = os.path.join(*os.path.normpath(folder).split(os.sep)[:-1])
            Eager._create_path(one_up)
            os.mkdir(folder)
