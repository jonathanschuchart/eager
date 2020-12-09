import os
from multiprocessing import Pool
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

    def predict(self, pairs, parallel=False) -> List[float]:
        if parallel:
            with Pool(6) as pool:
                chunks = [pairs[i:i+128] for i in range(0, len(pairs), 128)]
                result = pool.map(self._predict_pairs, chunks)
                return [pred for preds in result for pred in preds]
        else:
            return self._predict_pairs(pairs)

    def _predict_pair(self, e1: int, e2: int) -> float:
        return self._classifier.predict([self.pair_to_vec(e1, e2)])[0]

    def _predict_pairs(self, pairs: List[Tuple[int, int]]) -> List[float]:
        return self._classifier.predict([self.pair_to_vec(e[0], e[1]) for e in pairs])

    def evaluate(self, labelled_pairs: List[Tuple[int, int, int]]) -> EvalResult:
        return self.evaluate_against_gold(labelled_pairs, labelled_pairs)

    def evaluate_against_gold(
        self, pairs: List[Tuple[int, ...]], gold: List[Tuple[int, int, int]]
    ) -> EvalResult:
        prediction = self.predict(pairs)
        return self._eval.evaluate(
            gold, [(e[0], e[1], p) for p, e in zip(prediction, pairs)],
        )

    @staticmethod
    def _create_path(folder):
        if not os.path.exists(folder):
            one_up = os.path.join(*os.path.normpath(folder).split(os.sep)[:-1])
            Eager._create_path(one_up)
            os.mkdir(folder)
