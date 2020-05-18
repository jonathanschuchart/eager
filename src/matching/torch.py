import random
from typing import List, Tuple, Callable, Iterable

import torch
import numpy as np

from matching.eval import Eval
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
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, 1, gamma=0.95
        )
        self._criterion = torch.nn.CrossEntropyLoss()
        self._epochs = epochs
        self._batch_size = batch_size
        self._eval = Eval(self.predict_pair)

    def fit(
        self,
        labelled_train_pairs: List[Tuple[int, int, int]],
        labelled_val_pairs: List[Tuple[int, int, int]],
    ):
        data = self._batchify(labelled_train_pairs)
        val_data = self._batchify(labelled_val_pairs)
        outputs = None
        for epoch in range(0, self._epochs):
            loss, outputs = self._fit_epoch(data, val_data, epoch)
            print(f"training: {loss}")
            prediction = self.predict([(e[0], e[1]) for e in labelled_val_pairs])
            prediction = [
                (e[0], e[1]) for p, e in zip(prediction, labelled_val_pairs) if p > 0.5
            ]
            print(self._eval.evaluate(labelled_val_pairs, prediction))
        return outputs

    def _batchify(self, matching_pairs):
        random.Random().shuffle(matching_pairs)

        def make_batch_data(cur_batch):
            pair_labels = matching_pairs[cur_batch : cur_batch + self._batch_size]
            x = [self.pair_to_vec(e1, e2) for e1, e2, _ in pair_labels]
            y = [label for _, _, label in pair_labels]
            return x, y

        return [
            make_batch_data(i)
            for i in range(0, len(matching_pairs) - 1, self._batch_size)
        ]

    def _fit_epoch(self, data, val_data, epoch) -> Tuple[float, List[float]]:
        all_outputs = []
        total_loss = 0.0
        for x, y in data:
            loss, output = self._fit_batch(x, y)
            all_outputs.extend(output.detach().cpu().numpy())
            total_loss += loss
        with torch.no_grad():
            val_loss = 0
            for x, y in val_data:
                loss, _ = self._eval_batch(x, y)
                val_loss += loss
            print(f"validation: {val_loss / len(val_data)}")
        return total_loss / len(data), all_outputs

    def _fit_batch(self, x, y):
        self._optimizer.zero_grad()
        loss, output = self._eval_batch(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self._optimizer.step()
        return loss.item(), output

    def _eval_batch(self, x, y):
        output = self.model(torch.tensor(x))
        loss = self._criterion(output, torch.tensor(y))
        return loss, output

    def predict_pair(self, x1: int, x2: int) -> float:
        return self.predict([(x1, x2)])[0]

    def predict(self, pairs: List[Tuple[int, int]]) -> List[float]:
        with torch.no_grad():
            return (
                self.model(torch.tensor([self.pair_to_vec(x[0], x[1]) for x in pairs]))
                .detach()
                .cpu()
                .numpy()[:, 1]
            )
