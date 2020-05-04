import random
from typing import List, Tuple, Dict

import torch
import numpy as np


class TorchModelTrainer:
    def __init__(self, model: torch.nn.Module, epochs: int, batch_size: int):
        self.model = model
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, 1, gamma=0.95
        )
        self._criterion = torch.nn.CrossEntropyLoss()
        self._epochs = epochs
        self._batch_size = batch_size

    def _batchify(self, matching_pairs, embedding_lookup):
        random.Random().shuffle(matching_pairs)

        def make_batch_data(cur_batch):
            pair_labels = matching_pairs[cur_batch : cur_batch + self._batch_size]
            X = [
                [embedding_lookup[e1], embedding_lookup[e2]]
                for e1, e2, _ in pair_labels
            ]
            Y = [label for _, _, label in pair_labels]
            return X, Y

        return [
            make_batch_data(i)
            for i in range(0, len(matching_pairs) - 1, self._batch_size)
        ]

    def fit(
        self,
        labelled_train_pairs: List[Tuple[str, str, int]],
        labelled_val_pairs: List[Tuple[str, str, int]],
        embedding_lookup: Dict[str, np.array],
    ):
        data = self._batchify(labelled_train_pairs, embedding_lookup)
        val_data = self._batchify(labelled_val_pairs, embedding_lookup)
        outputs = None
        for epoch in range(0, self._epochs):
            loss, outputs = self._fit_epoch(data, val_data, epoch)
            print(f"training: {loss}")
        return outputs

    def _fit_epoch(self, data, val_data, epoch):
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

    def predict(
        self, pairs: List[Tuple[str, str, int]], embedding_lookup: Dict[str, np.array]
    ):
        x = torch.tensor(
            [[embedding_lookup[e[0]], embedding_lookup[e[0]]] for e in pairs]
        )
        with torch.no_grad():
            return self.model(x).detach().cpu().numpy()
