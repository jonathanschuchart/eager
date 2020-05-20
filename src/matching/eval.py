from typing import List, Tuple, Dict, Iterable, Callable

from sklearn.neighbors import KNeighborsTransformer
import numpy as np


class EvalResult:
    def __init__(
        self,
        precision: float,
        recall: float,
        f1: float,
        hits_at: Dict[int, float],
        mrr: float,
        mr: float,
    ):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.hits_at = hits_at
        self.mrr = mrr
        self.mr = mr

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


class Eval:
    def __init__(self, pair_similarity: Callable[[int, int], float]):
        self.pair_similarity = pair_similarity

    def evaluate(
        self, gold: List[Tuple[int, int, int]], prediction: Iterable[Tuple[int, int]]
    ) -> EvalResult:
        prec, recall, f1 = self._prec_rec_f1(gold, prediction)
        # hits_at, mrr, mr = self._rank_eval(gold)
        hits_at, mrr, mr = {}, 0, 0

        return EvalResult(prec, recall, f1, hits_at, mrr, mr)

    @staticmethod
    def _prec_rec_f1(
        gold: List[Tuple[int, int, int]], prediction: Iterable[Tuple[int, int]]
    ):
        gold_pos = {t[:2] for t in gold if t[2] == 1}
        gold_neg = {t[:2] for t in gold if t[2] == 0}
        prediction = set(prediction)

        true_pos = len(prediction & gold_pos)
        false_pos = len(prediction & gold_neg)
        false_neg = len(gold_pos - prediction)

        prec = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0
        return prec, recall, f1

    def _rank_eval(self, gold: List[Tuple[int, int, int]]):
        neigh = KNeighborsTransformer(
            mode="distance",
            n_neighbors=10,
            metric=lambda x1, x2: self.pair_similarity(x1[0], x2[0]),
            n_jobs=-1,
        )
        entities = np.asarray(
            list({e[0] for e in gold} | {e[1] for e in gold}), dtype=np.int
        ).reshape(-1, 1)
        neigh.fit(entities)
        neigh_dist, neigh_ind = neigh.kneighbors(entities, return_distance=True)
        counts = {1: 0, 5: 0, 10: 0, 50: 0}
        ranks = [float("inf") for _ in gold]
        for gold_index, (e1, e2, label) in enumerate(gold):
            if label != 1:
                continue
            for n, neigh in enumerate(neigh_ind[e1]):
                if neigh == e2:
                    for k in counts.keys():
                        if n < k:
                            counts[k] += 1
                    ranks[gold_index] = n
        hits_at = {k: v / len(gold) for k, v in counts}
        mrr = 1.0 / len(gold) * sum(1.0 / r for r in ranks)
        mr = 1.0 / len(gold) * sum(r for r in ranks if r < float("inf"))
        return hits_at, mrr, mr
