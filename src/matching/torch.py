import random
from typing import List, Tuple, Callable, Iterable, Union

import torch
import numpy as np

from matching.eval import Eval, EvalResult
from matching.matcher import MatchModelTrainer


class TorchModelTrainer(MatchModelTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int,
        batch_size: int,
        pair_to_vec: Callable[[int, int], np.ndarray],
    ):
        self.pair_to_vec = pair_to_vec
        self.model = model
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=0.25)
        self._criterion = torch.nn.CrossEntropyLoss()
        # self._criterion = torch.nn.BCELoss()
        self._epochs = epochs
        self._batch_size = batch_size
        self._eval = Eval(self.predict_pair)

    def fit(
        self,
        labelled_train_pairs: List[Tuple[int, int, int]],
        labelled_val_pairs: List[Tuple[int, int, int]],
    ):
        rnd = random.Random()
        rnd.shuffle(labelled_train_pairs)
        rnd.shuffle(labelled_val_pairs)
        data = self._batchify(labelled_train_pairs)
        val_data = self._batchify(labelled_val_pairs)
        outputs = None
        for epoch in range(0, self._epochs):
            train_loss, outputs = self._fit_epoch(data, epoch)
            train_prediction = [
                e[:2] for p, e in zip(outputs, labelled_train_pairs) if p[1] > 0.5
            ]
            train_eval = self._eval.evaluate(labelled_train_pairs, train_prediction)
            val_loss, val_prediction_ = self._eval_data(val_data)
            val_prediction = [
                e[:2] for p, e in zip(val_prediction_, labelled_val_pairs) if p[1] > 0.5
            ]
            print(f"epoch: {epoch}")
            print(f"training loss: {train_loss}, {train_eval}")
            val_eval = self._eval.evaluate(labelled_val_pairs, val_prediction)
            print(f"validation loss: {val_loss}, {val_eval}")
        return outputs

    def _batchify(self, matching_pairs):
        def make_batch_data(cur_batch):
            pair_labels = matching_pairs[cur_batch : cur_batch + self._batch_size]
            x = [self.pair_to_vec(e1, e2) for e1, e2, _ in pair_labels]
            y = [label for _, _, label in pair_labels]
            return x, y

        return [
            make_batch_data(i)
            for i in range(0, len(matching_pairs) - 1, self._batch_size)
        ]

    def _fit_epoch(self, data, epoch) -> Tuple[float, List[float]]:
        all_outputs = []
        total_loss = 0.0
        self.model.train()
        for x, y in data:
            loss, output = self._fit_batch(x, y)
            all_outputs.extend(output)
            total_loss += loss
        return total_loss / len(data), all_outputs

    def _eval_data(self, val_data):
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            outputs = []
            for x, y in val_data:
                loss, out = self._eval_batch(x, y)
                val_loss += loss
                outputs.extend(out)
            return val_loss / len(val_data), outputs

    def _fit_batch(self, x, y):
        self._optimizer.zero_grad()
        loss, output = self._eval_batch(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self._optimizer.step()
        return loss.item(), output

    def _eval_batch(self, x, y):
        output = self.model(torch.tensor(x, dtype=torch.float32))
        loss = self._criterion(output, torch.tensor(y, dtype=torch.long))
        return loss, output.detach().cpu().numpy()

    def predict_pair(self, x1: int, x2: int) -> float:
        return self.predict([(x1, x2)])[0]

    def predict(self, pairs: List[Tuple[int, ...]]) -> List[float]:
        with torch.no_grad():
            return (
                self.model(
                    torch.tensor(
                        [self.pair_to_vec(x[0], x[1]) for x in pairs],
                        dtype=torch.float32,
                    )
                )
                .detach()
                .cpu()
                .numpy()[1]
            )

    def evaluate(self, labelled_pairs: List[Tuple[int, int, int]]) -> EvalResult:
        data = self._batchify(labelled_pairs)
        loss, prediction_ = self._eval_data(data)
        prediction = [e[:2] for p, e in zip(prediction_, labelled_pairs) if p[1] > 0.5]
        return self._eval.evaluate(labelled_pairs, prediction)
