import random
from typing import List, Tuple

import numpy as np
from openea.modules.load.kgs import read_kgs_from_folder

from matching.mlp import MLP
from matching.torch import TorchModelTrainer
from similarity.similarities import (
    calculate_from_embeddings,
    calculate_from_embeddings_with_training,
    calculate_on_demand,
)


def sample_negative(pos_samples: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    negative_pairs = set()
    rnd = random.Random()
    entities_left = list({e[0] for e in pos_samples})
    entities_right = list({e[1] for e in pos_samples})
    pos_set = set(pos_samples)
    while len(negative_pairs) < len(pos_samples):
        e1 = rnd.choice(entities_left)
        e2 = rnd.choice(entities_right)
        if e1 != e2 and (e1, e2) not in pos_set and (e1, e2) not in negative_pairs:
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

    train_neg_samples = sample_negative(kgs.train_links)
    val_neg_samples = sample_negative(kgs.valid_links)
    train_links = train_neg_samples + [(e[0], e[1], 1) for e in kgs.train_links]
    random.shuffle(train_links)
    valid_links = val_neg_samples + [(e[0], e[1], 1) for e in kgs.valid_links]
    random.shuffle(valid_links)
    # test_links = neg_samples[int(0.9 * len(neg_samples)) :] + [
    #     (e[0], e[1], 1) for e in kgs.test_links
    # ]

    all_sims = calculate_from_embeddings_with_training(
        embeddings, train_links, kgs, "euclidean"
    )
    all_keys = {k for v in all_sims.values() for k in v.keys()}

    def pair_to_vec(e1: int, e2: int):
        sim = calculate_on_demand(embeddings, (e1, e2), kgs, "euclidean")
        sim_vec = [sim.get(k, 0) for k in all_keys]
        return np.concatenate([sim_vec, embeddings[int(e1)], embeddings[int(e2)]])

    model = MLP([len(all_keys) + 200, 50])
    epochs = 100
    batch_size = 1000
    model_trainer = TorchModelTrainer(model, epochs, batch_size, pair_to_vec)
    model_trainer.fit(train_links, valid_links)


if __name__ == "__main__":
    main()
