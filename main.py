import random
from typing import List, Tuple

import numpy as np
from openea.modules.load.kgs import read_kgs_from_folder

from matching.mlp import MLP
from matching.torch import TorchModelTrainer
from similarity.similarities import calculate_from_embeddings


def sample_negative(pos_samples: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    negative_pairs = set()
    rnd = random.Random()
    entities = list({e[0] for e in pos_samples} | {e[1] for e in pos_samples})
    pos_set = set(pos_samples)
    while len(negative_pairs) < len(pos_samples):
        e1 = rnd.choice(entities)
        e2 = rnd.choice(entities)
        if (
            e1 != e2
            and (e1, e2) not in pos_set
            and (e2, e1) not in pos_set
            and (e1, e2) not in negative_pairs
            and (e2, e1) not in negative_pairs
        ):
            negative_pairs.add((e1, e2))
    return [(e0, e1, 0) for e0, e1 in negative_pairs]


def main():
    embeddings = np.load(
        "/home/jonathan/output/results/GCN_Align/datasets/721_5fold20200515130138/ent_embeds.npy"
    )
    training_data_folder = "../datasets/D_W_15K_V1/"
    kgs = read_kgs_from_folder(
        training_data_folder, "721_5fold/1/", "sharing", ordered=False
    )

    model = MLP([200, 50])
    epochs = 10
    batch_size = 1000

    # TODO: Add similarity vector to concatenation
    def pair_to_vec(e1: int, e2: int):
        return np.concatenate([embeddings[e1], embeddings[e2]])

    model_trainer = TorchModelTrainer(model, epochs, batch_size, pair_to_vec)

    neg_samples = sample_negative(kgs.train_links + kgs.valid_links + kgs.test_links)
    train_links = neg_samples[: int(0.7 * len(neg_samples))] + [
        (e[0], e[1], 1) for e in kgs.train_links
    ]
    valid_links = neg_samples[
        int(0.7 * len(neg_samples)) : int(0.9 * len(neg_samples))
    ] + [(e[0], e[1], 1) for e in kgs.valid_links]
    test_links = neg_samples[int(0.9 * len(neg_samples)) :] + [
        (e[0], e[1], 1) for e in kgs.test_links
    ]
    model_trainer.fit(train_links, valid_links)


if __name__ == "__main__":
    main()
