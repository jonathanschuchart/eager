import time
from multiprocessing import Pool
from typing import List, Tuple

from dataset.dataset import Dataset
from eager import Eager


class Experiment:
    def __init__(self, eager: Eager):
        self.model = eager

    def run_with_queries(self, dataset: Dataset, query_list: List[Tuple[int, int]]):
        print("starting training")
        start = time.time()
        self.model.fit(dataset.labelled_train_pairs, dataset.labelled_val_pairs)
        train_time = time.time() - start
        print("finished training")
        start = time.time()
        predictions = [(q[0], q[1], float(p)) for q, p in zip(query_list, self.model.predict(query_list, parallel=True))]
        import itertools
        counts = [(k, len(list(v))) for k, v in itertools.groupby([p for p in predictions if p[2] > 0.5], lambda x: x[0])]
        sorted_counts = sorted(counts, key=lambda x: x[1])
        count_counts = [(k, len(list(v))) for k, v in itertools.groupby(sorted_counts, lambda x: x[1])]
        print(f"count of entity match counts: {count_counts}")
        pred_time = time.time() - start
        print(f"finished predictions on knn. Took {pred_time}s")
        with open(f"predictions-{self.model}-{dataset.name().replace('/', '-')}.json", "w") as f:
            import json
            json.dump(predictions, f)
        pred_dict = {(p[0], p[1]): 2 for p in predictions}
        print(len(predictions))
        print(len(pred_dict))
        train_predictions = [
            (p[0], p[1], pred_dict.get((p[0], p[1]), 0))
            for p in dataset.labelled_train_pairs
        ]
        print(len(train_predictions))
        val_predictions = [
            (p[0], p[1], pred_dict.get((p[0], p[1]), 0))
            for p in dataset.labelled_val_pairs
        ]
        print(len(val_predictions))
        train_eval = self.model._eval.evaluate(
            dataset.labelled_train_pairs, train_predictions,
        )
        print(f"{self.model} - train: {train_eval}")

        valid_eval = self.model._eval.evaluate(
            dataset.labelled_val_pairs, val_predictions,
        )
        print(f"{self.model} - valid: {valid_eval}")
        start = time.time()
        test_eval = self.model._eval.evaluate(dataset.labelled_test_pairs, predictions)
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

    def run(self, dataset: Dataset):
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
    def __init__(
        self,
        dest_folder: str,
        experiments: List[Experiment],
        dataset: Dataset,
        query_list: List[Tuple[int, int]],
    ):
        self.dest_folder = dest_folder
        self.experiments = experiments
        self.dataset = dataset
        self.query_list = query_list

    def run(self):
        num_experiments = len(self.experiments)
        if self.query_list is None:
            with Pool() as pool:
                return pool.starmap(
                    Experiment.run,
                    zip(self.experiments, [self.dataset] * num_experiments),
                )
        else:
            # with Pool() as pool:
            #     return pool.starmap(
            #         Experiment.run_with_queries,
            #         zip(
            #             self.experiments,
            #             [self.dataset] * num_experiments,
            #             [self.query_list] * num_experiments,
            #         ),
            #     )
            return [e.run_with_queries(self.dataset, self.query_list) for e in self.experiments]
