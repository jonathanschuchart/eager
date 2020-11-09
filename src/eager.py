import json
import os

import joblib

from similarity.create_training import create_feature_similarity_frame


class Eager:
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer

    def train(self, train_pairs, val_pairs):
        self.model_trainer.fit(train_pairs, val_pairs)

    def predict(self, pairs):
        return self.model_trainer.evaluate(pairs)

    def save(self, folder):
        if any(a is None for a in [self.all_sims, self.min_max, self.scale_cols]):
            raise Exception("Trying to save untrained Model")
        Eager._create_path(folder)
        self.all_sims.to_parquet(os.path.join(folder, "all_sims.parquet"))
        joblib.dump(self.min_max, os.path.join(folder, "min_max.pkl"))
        with open(os.path.join(folder, "scale_cols.json"), "w") as f:
            json.dump(self.scale_cols, f)
        # self.model_trainer

    @staticmethod
    def _create_path(folder):
        if not os.path.exists(folder):
            one_up = os.path.join(*os.path.normpath(folder).split(os.sep)[:-1])
            Eager._create_path(one_up)
            os.mkdir(folder)

    def load(self, folder):
        pass
