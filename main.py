import random
from typing import List, Tuple

import numpy as np
from openea.approaches import GCN_Align
from openea.modules.args.args_hander import load_args
from openea.modules.load.kgs import read_kgs_from_folder
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from matching.matcher import MatchModelTrainer
from matching.mlp import MLP
from matching.sklearn import SkLearnMatcher
from matching.torch import TorchModelTrainer
from similarity.similarities import (
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
        if (
            e1 != e2
            and (e1, e2, 1) not in pos_set
            and (e2, e1, 1) not in pos_set
            and (e1, e2) not in negative_pairs
            and (e2, e1) not in negative_pairs
        ):
            negative_pairs.add((e1, e2))
    return [(e0, e1, 0) for e0, e1 in negative_pairs]


def main():
    embeddings = np.load(
        "/home/jonathan/output/results/GCN_Align/datasets/721_5fold20200526171142/ent_embeds.npy"
    )
    training_data_folder = "../datasets/D_W_15K_V1/"
    kgs = read_kgs_from_folder(
        training_data_folder, "721_5fold/1/", "mapping", ordered=False
    )

    # model = GCN_Align()
    # model.set_args(load_args("../OpenEA/run/args/gcnalign_args_15K.json"))
    # model.set_kgs(kgs)
    # model.init()
    # model.run()
    # model.save()

    train_neg_samples = sample_negative(kgs.train_links)
    val_neg_samples = sample_negative(kgs.valid_links)
    train_links = train_neg_samples + [(e[0], e[1], 1) for e in kgs.train_links]
    valid_links = val_neg_samples + [(e[0], e[1], 1) for e in kgs.valid_links]

    all_sims = calculate_from_embeddings_with_training(
        embeddings, train_links, kgs, "euclidean"
    )
    all_keys = {k for v in all_sims.values() for k in v.keys()}

    sim_cnt = [0 for _ in all_keys]

    def pair_to_vec(e1: int, e2: int):
        sim = calculate_on_demand(embeddings, (e1, e2), kgs, "euclidean")
        sim_vec = np.asarray([sim.get(k, 0) for k in all_keys])
        norm = np.linalg.norm(sim_vec)
        sim_vec = sim_vec / (norm if norm > 0 else 1)
        return np.concatenate([sim_vec, embeddings[int(e1)], embeddings[int(e2)]])

    # model = MLP([len(all_keys) + 200, 500])
    # epochs = 400
    # batch_size = 1000
    # model_trainer = TorchModelTrainer(model, epochs, batch_size, pair_to_vec)
    print("random forest 20")
    model_trainer = SkLearnMatcher(pair_to_vec, RandomForestClassifier(20))
    run(model_trainer, train_links, valid_links)

    print("random forest 50")
    model_trainer = SkLearnMatcher(pair_to_vec, RandomForestClassifier(50))
    run(model_trainer, train_links, valid_links)

    print("random forest 100")
    model_trainer = SkLearnMatcher(pair_to_vec, RandomForestClassifier(100))
    run(model_trainer, train_links, valid_links)

    print("random forest 200")
    model_trainer = SkLearnMatcher(pair_to_vec, RandomForestClassifier(200))
    run(model_trainer, train_links, valid_links)

    print("random forest 500")
    model_trainer = SkLearnMatcher(pair_to_vec, RandomForestClassifier(500))
    run(model_trainer, train_links, valid_links)

    print("decision tree")
    model_trainer = SkLearnMatcher(pair_to_vec, DecisionTreeClassifier())
    run(model_trainer, train_links, valid_links)

    print("svc")
    model_trainer = SkLearnMatcher(pair_to_vec, svm.SVC())
    run(model_trainer, train_links, valid_links)


def run(model_trainer: MatchModelTrainer, train_links, valid_links):
    model_trainer.fit(train_links, valid_links)
    print(f"training: {model_trainer.evaluate(train_links)}")
    print(f"validation: {model_trainer.evaluate(valid_links)}")


if __name__ == "__main__":
    main()
