from multiprocessing import Pool
from typing import List, Tuple, Dict, Iterable, Callable

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
        prediction: Iterable[Tuple[int, int, int, float]],
    ):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.hits_at = hits_at
        self.mrr = mrr
        self.mr = mr
        self.prediction = prediction

    def __str__(self):
        return {k: v for k, v in self.__dict__.items() if k != "prediction"}.__str__()

    def __repr__(self):
        return {k: v for k, v in self.__dict__.items() if k != "prediction"}.__repr__()


class Eval:
    def __init__(self, pair_similarity: Callable[[int, int], float]):
        self.pair_similarity = pair_similarity

    def evaluate(
        self,
        gold: List[Tuple[int, int, int]],
        prediction: Iterable[Tuple[int, int, int, float]],
    ) -> EvalResult:
        prec, recall, f1 = self._prec_rec_f1(
            gold, (p[:2] for p in prediction if p[3] > 0.5)
        )
        # hits_at, mrr, mr = self._rank_eval(gold)
        hits_at, mrr, mr = {}, 0, 0

        return EvalResult(prec, recall, f1, hits_at, mrr, mr, prediction)

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
        # neigh = KNeighborsTransformer(
        #     mode="distance",
        #     n_neighbors=10,
        #     metric=lambda x1, x2: self.pair_similarity(x1[0], x2[0]),
        #     n_jobs=-1,
        # )
        # entities = np.asarray(
        #     list({e[0] for e in gold} | {e[1] for e in gold}), dtype=np.int
        # ).reshape(-1, 1)
        # neigh.fit(entities)
        # neigh_dist, neigh_ind = neigh.kneighbors(entities, return_distance=True)
        similarities, left_entities, right_entities = self.sim_mat(gold)
        left_ent_to_id = {e: i for i, e in enumerate(left_entities)}
        right_ent_to_id = {e: i for i, e in enumerate(right_entities)}
        counts = {1: 0, 5: 0, 10: 0, 50: 0}
        ranks = np.asarray([float("inf") for _ in gold])
        gold_lists = {g[0]: [e[1] for e in gold if e[0] == g[0]] for g in gold}
        gold = sorted(gold)
        sorted_sims = np.argsort(similarities, axis=1)[:, ::-1]
        for r_ind, (left, right, label) in enumerate(gold):
            if label != 1:
                continue
            left_idx = left_ent_to_id[left]
            right_idx = right_ent_to_id[right]
            rank_idx = np.where(sorted_sims[left_idx] == right_idx)[0]
            for k in counts.keys():
                if rank_idx < k:
                    counts[k] += 1
            ranks[r_ind] = rank_idx
        hits_at = {k: v / len(gold) for k, v in counts}
        mrr = 1.0 / len(gold) * sum(1.0 / r for r in ranks)
        mr = 1.0 / len(gold) * sum(r for r in ranks if r < float("inf"))
        return hits_at, mrr, mr

    def sim_mat(
        self, gold: List[Tuple[int, int, int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_entities = np.sort(np.asarray(list({e[0] for e in gold}), dtype=np.int))
        right_entities = np.sort(np.asarray(list({e[1] for e in gold}), dtype=np.int))

        # sims = np.zeros((left_entities.shape[0], right_entities.shape[0]))

        with Pool() as pool:
            sims = pool.starmap(
                self.pair_similarity,
                ((x, y) for x in left_entities for y in right_entities),
            )
        # for l_idx, left in enumerate(left_entities):
        #     start = time.time()
        #     for r_idx, right in enumerate(right_entities):
        #         sims[l_idx, r_idx] = self.pair_similarity(left, right)
        #     print(f"finished {l_idx}: {time.time() - start}")
        print("finished similarity matrix")
        sims = np.asarray(sims).reshape((len(left_entities), len(right_entities)))
        np.save("output/sim_mat.np", sims)
        np.save("output/left.np", left_entities)
        np.save("output/right.np", right_entities)
        return sims, left_entities, right_entities
