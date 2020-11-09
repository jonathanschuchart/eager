import time
from typing import List, Callable

from matching.classifiers import SkLearnMatcher
from matching.matcher import MatchModelTrainer


class Experiment:
    def __init__(self, model_factory: Callable[[], SkLearnMatcher]):
        self.model_factory = model_factory

    def run(self, dataset):
        model = self.model_factory()
        start = time.time()
        model.fit(dataset.labelled_train_pairs, dataset.labelled_val_pairs)
        train_time = time.time() - start
        train_eval = model.evaluate(dataset.labelled_train_pairs)
        print(f"train: {train_eval}")
        valid_eval = model.evaluate(dataset.labelled_val_pairs)
        print(f"valid: {valid_eval}")
        start = time.time()
        test_eval = model.evaluate(dataset.labelled_test_pairs)
        test_time = time.time() - start
        print(f"test: {test_eval}")

        return {
            "model_name": model.__str__(),
            "vector_name": model.pair_to_vec.name,
            "train_precision": train_eval.precision,
            "train_recall": train_eval.recall,
            "train_f1": train_eval.f1,
            "train_prediction": train_eval.prediction,
            "val_precision": valid_eval.precision,
            "val_recall": valid_eval.recall,
            "val_f1": valid_eval.f1,
            "val_prediction": valid_eval.prediction,
            "test_precision": test_eval.precision,
            "test_recall": test_eval.recall,
            "test_f1": test_eval.f1,
            "test_prediction": test_eval.prediction,
            "train_time": train_time,
            "test_time": test_time,
        }


class Experiments:
    def __init__(self, dest_folder, experiments: List[Experiment], dataset):
        self.dest_folder = dest_folder
        self.experiments = experiments
        self.dataset = dataset

    def run(self):
        # TODO: parallelize
        return [e.run(self.dataset) for e in self.experiments]
