import json
from os import path
from glob import glob
from typing import Any, Dict

import numpy as np
from openea.approaches import GCN_Align, BootEA
from openea.models.basic_model import BasicModel
from openea.modules.args.args_hander import load_args
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from dataset.csv_dataset import CsvDataset, CsvType
from dataset.dataset import Dataset
from dataset.openea_dataset import OpenEaDataset
from dataset.scads_dataset import ScadsDataset
from matching.matcher import MatchModelTrainer
from matching.pair_to_vec import (
    SimAndEmb,
    SimAndEmbNormalized,
    OnlySim,
    OnlySimNormalized,
    OnlyEmb,
    PairToVec,
)
from matching.sklearn import SkLearnMatcher
from similarity.create_training import create_feature_similarity_frame
import pandas as pd

models = [
    lambda pair_to_vec: SkLearnMatcher(pair_to_vec, svm.SVC(), "svc"),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(20), "random forest 20"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(50), "random forest 50"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(100), "random forest 100"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(200), "random forest 200"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(500), "random forest 500"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, DecisionTreeClassifier(), "decision tree"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, GaussianNB(), "gaussian naive bayes"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec,
        MLPClassifier(
            solver="adam", alpha=1e-5, hidden_layer_sizes=(200, 20), max_iter=500,
        ),
        "MLP",
    ),
]


def gcn_align_15(dataset_name: str):
    model = GCN_Align()
    args = load_args("../OpenEA/run/args/gcnalign_args_15K.json")
    args.output = f"output/results/{dataset_name}/"
    model.set_args(args)
    return model, "gcn_align_15", lambda m: m.vec_se


def boot_ea_15(dataset_name: str):
    model = BootEA()
    args = load_args("../OpenEA/run/args/bootea_args_15K.json")
    args.output = f"output/results/{dataset_name}/"
    model.set_args(args)
    return model, "boot_ea_15", lambda m: m.ent_embeds.eval(session=m.session)


def gcn_align_100(dataset_name: str):
    model = GCN_Align()
    args = load_args("../OpenEA/run/args/gcnalign_args_100K.json")
    args.output = f"output/results/{dataset_name}"
    model.set_args(args)
    return model, "gcn_align_100", lambda m: m.vec_se


datasets = [
    lambda: (
        OpenEaDataset("../datasets/D_W_15K_V1/", "721_5fold/1/"),
        boot_ea_15("D_W_15K_V1"),
    ),
    lambda: (
        OpenEaDataset("../datasets/D_W_15K_V1/", "721_5fold/1/"),
        gcn_align_15("D_W_15K_V1"),
    ),
    # lambda: (
    #     OpenEaDataset("../datasets/D_W_100K_V1/", "721_5fold/1/"),
    #     gcn_align_100("D_W_100K_V1"),
    # ),
    lambda: (
        ScadsDataset("data/ScadsMB/100", "imdb", "tmdb"),
        gcn_align_15("ScadsMB_imdb_tmdb"),
    ),
    lambda: (
        ScadsDataset("data/ScadsMB/100", "imdb", "tmdb"),
        boot_ea_15("ScadsMB_imdb_tmdb"),
    ),
    lambda: (
        ScadsDataset("data/ScadsMB/100", "imdb", "tvdb"),
        gcn_align_15("ScadsMB_imdb_tvdb"),
    ),
    lambda: (
        ScadsDataset("data/ScadsMB/100", "imdb", "tvdb"),
        boot_ea_15("ScadsMB_imdb_tvdb"),
    ),
    lambda: (
        ScadsDataset("data/ScadsMB/100", "tmdb", "tvdb"),
        gcn_align_15("ScadsMB_tmdb_tvdb"),
    ),
    lambda: (
        ScadsDataset("data/ScadsMB/100", "tmdb", "tvdb"),
        gcn_align_15("ScadsMB_tmdb_tvdb"),
    ),
    lambda: (
        ScadsDataset("data/ScadsMB/100", "tmdb", "tvdb"),
        boot_ea_15("ScadsMB_tmdb_tvdb"),
    ),
    lambda: (
        CsvDataset(
            CsvType.products,
            "data/amazon-google/Amazon.csv",
            "data/amazon-google/GoogleProducts.csv",
            "data/amazon-google/Amzon_GoogleProducts_perfectMapping.csv",
        ),
        gcn_align_15("Amazon_Google"),
    ),
    lambda: (
        CsvDataset(
            CsvType.products,
            "data/amazon-google/Amazon.csv",
            "data/amazon-google/GoogleProducts.csv",
            "data/amazon-google/Amzon_GoogleProducts_perfectMapping.csv",
        ),
        boot_ea_15("Amazon_Google"),
    ),
    lambda: (
        CsvDataset(
            CsvType.articles,
            "data/dblp-acm/DBLP2.csv",
            "data/dblp-acm/ACM.csv",
            "data/dblp-acm/DBLP-ACM_perfectMapping.csv",
        ),
        gcn_align_15("DBLP_ACM"),
    ),
    lambda: (
        CsvDataset(
            CsvType.articles,
            "data/dblp-acm/DBLP2.csv",
            "data/dblp-acm/ACM.csv",
            "data/dblp-acm/DBLP-ACM_perfectMapping.csv",
        ),
        boot_ea_15("DBLP_ACM"),
    ),
]


def find_existing_embedding(embedding_model: BasicModel):
    folder = embedding_model.out_folder
    folder = "/".join(folder.split("/")[:-2])

    dirs = sorted(
        [d for d in glob(f"{folder}/*") if embedding_model.args.dataset_division in d]
    )
    if any(dirs):
        return dirs[-1]
    return None


def run_for_dataset(dataset_idx):
    import tensorflow as tf

    tf.reset_default_graph()
    dataset, (embedding_model, embedding_name, embedding_fn) = datasets[dataset_idx]()
    dataset.add_negative_samples()

    kgs = dataset.kgs()

    existing_folder = find_existing_embedding(embedding_model)
    if existing_folder is not None:
        embeddings = np.load(f"{existing_folder}/ent_embeds.npy")
    else:
        embedding_model.set_kgs(kgs)
        embedding_model.init()
        embedding_model.run()
        embedding_model.test()
        embedding_model.save()
        embeddings = embedding_fn(embedding_model)

    all_pairs = (
        dataset.labelled_train_pairs
        + dataset.labelled_val_pairs
        + dataset.labelled_test_pairs
    )
    if existing_folder is not None and path.exists(f"{existing_folder}/all_sims.csv"):
        all_sims = pd.read_csv(f"{existing_folder}/all_sims.csv")
        with open(f"{existing_folder}/min_max.json") as f:
            min_max = json.load(f)
        min_max = MinMaxScaler().set_params(**min_max)
    else:
        all_sims, min_max = create_feature_similarity_frame(
            embeddings, all_pairs, kgs, only_training=True
        )
        with open(f"{existing_folder}/min_max.json", "w") as f:
            json.dump(min_max.get_params(), f)
        output_folder = existing_folder or embedding_model.out_folder[:-1]
        all_sims.to_csv(f"{output_folder}/all_sims.csv")
    all_sims = all_sims.dropna(axis=1, how="all", thresh=int(0.1 * len(all_sims)))

    pair_to_vecs = [
        SimAndEmb(embeddings, all_sims, min_max, kgs),
        SimAndEmbNormalized(embeddings, all_sims, min_max, kgs),
        OnlySim(embeddings, all_sims, min_max, kgs),
        OnlySimNormalized(embeddings, all_sims, min_max, kgs),
        OnlyEmb(embeddings, all_sims, min_max, kgs),
    ]
    with open(f"output/{dataset.name()}-{embedding_name}", "w") as file_to_print:
        results_list = []
        for pair_to_vec in pair_to_vecs:
            for model in models:
                model = model(pair_to_vec)
                print(f"\n{model} - {type(pair_to_vec).__name__}", file=file_to_print)
                results_list.append(run(model, dataset, pair_to_vec, file_to_print))
                file_to_print.flush()
    results = pd.DataFrame(
        data=results_list,
        columns=[
            "model_name",
            "vector_name",
            "train_precision",
            "train_recall",
            "train_f1",
            "val_precision",
            "val_recall",
            "val_f1",
            "test_precision",
            "test_recall",
            "test_f1",
        ],
    )
    results.to_csv(f"{dataset.name()}-{embedding_name}.csv", index=False)


def main():
    # embeddings = np.load(
    #     "/home/jonathan/output/results/GCN_Align/datasets/721_5fold20200526171142/ent_embeds.npy"
    # )

    # model = MLP([len(all_keys) + 200, 500])
    # epochs = 400
    # batch_size = 1000
    # model_trainer = TorchModelTrainer(model, epochs, batch_size, pair_to_vec)
    #
    # with Pool() as pool:
    #     pool.map(run_for_dataset, range(len(datasets)))
    for i in range(len(datasets)):
        run_for_dataset(i)
    # run_for_dataset(-1)


def run(
    model_trainer: MatchModelTrainer, dataset: Dataset, pair_to_vec: PairToVec, file
) -> Dict[str, Any]:
    model_trainer.fit(dataset.labelled_train_pairs, dataset.labelled_val_pairs)
    train_eval = model_trainer.evaluate(dataset.labelled_train_pairs)
    print(f"train: {train_eval}", file=file)
    valid_eval = model_trainer.evaluate(dataset.labelled_val_pairs)
    print(f"valid: {valid_eval}", file=file)
    test_eval = model_trainer.evaluate(dataset.labelled_test_pairs)
    print(f"test: {test_eval}", file=file)
    return {
        "model_name": model_trainer.__str__(),
        "vector_name": type(pair_to_vec).__name__,
        "train_precision": train_eval.precision,
        "train_recall": train_eval.recall,
        "train_f1": train_eval.f1,
        "val_precision": valid_eval.precision,
        "val_recall": valid_eval.recall,
        "val_f1": valid_eval.f1,
        "test_precision": test_eval.precision,
        "test_recall": test_eval.recall,
        "test_f1": test_eval.f1,
    }


if __name__ == "__main__":
    main()
