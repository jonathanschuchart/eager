import time

from dataset.dataset import Dataset
from eager import Eager


class Experiment:
    def __init__(self, eager: Eager):
        self.model = eager

    def run(self, dataset: Dataset):
        print(f"{dataset.name()} - {self.model} - starting training")
        start = time.time()
        self.model.fit(dataset.labelled_train_pairs, dataset.labelled_val_pairs)
        train_time = time.time() - start
        print(f"{dataset.name()} - {self.model} - finished training")
        train_eval = self.model.evaluate(dataset.labelled_train_pairs)
        print(f"{dataset.name()} - {self.model} - train: {train_eval}")
        valid_eval = self.model.evaluate(dataset.labelled_val_pairs)
        print(f"{dataset.name()} - {self.model} - valid: {valid_eval}")
        start = time.time()
        test_eval = self.model.evaluate(dataset.labelled_test_pairs)
        test_time = time.time() - start
        print(f"{dataset.name()} - {self.model} - test: {test_eval}")

        results = {
            "model_name": self.model.__str__(),
            "vector_name": self.model.pair_to_vec.name,
            "train_precision": train_eval.precision,
            "train_recall": train_eval.recall,
            "train_f1": train_eval.f1,
            "val_precision": valid_eval.precision,
            "val_recall": valid_eval.recall,
            "val_f1": valid_eval.f1,
            "test_precision": test_eval.precision,
            "test_recall": test_eval.recall,
            "test_f1": test_eval.f1,
            "train_time": train_time,
            "test_time": test_time,
        }

        artifacts = {
            "train_prediction": train_eval.prediction,
            "val_prediction": valid_eval.prediction,
            "test_prediction": test_eval.prediction,
        }
        return results, artifacts
