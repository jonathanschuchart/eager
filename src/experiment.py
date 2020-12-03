import time
from multiprocessing import Pool
from typing import List

from eager import Eager


class Experiment:
    def __init__(self, eager: Eager):
        self.model = eager

    def run(self, dataset):
        print("starting training")
        start = time.time()
        self.model.fit(dataset.labelled_train_pairs, dataset.labelled_val_pairs)
        train_time = time.time() - start
        print("finished training")
        train_eval = self.model.evaluate(dataset.labelled_train_pairs)
        print(f"{self.model} - train: {train_eval}")
        valid_eval = self.model.evaluate(dataset.labelled_val_pairs)
        print(f"{self.model} - valid: {valid_eval}")
        start = time.time()
        test_eval = self.model.evaluate(dataset.labelled_test_pairs)
        test_time = time.time() - start
        print(f"{self.model} - test: {test_eval}")

        return {
            "model_name": self.model.__str__(),
            "vector_name": self.model.pair_to_vec.name,
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
        num_experiments = len(self.experiments)
        with Pool() as pool:
            return pool.starmap(
                Experiment.run, zip(self.experiments, [self.dataset] * num_experiments)
            )
        # return [e.run(self.dataset) for e in self.experiments]
