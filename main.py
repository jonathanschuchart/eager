import random
from glob import glob
from os import path

import numpy as np
from openea.models.basic_model import BasicModel
from sentence_transformers import SentenceTransformer

from attribute_features import CartesianCombination, AllToOneCombination, _remove_type
from dataset.dataset import Dataset
from distance_measures import DateDistance, EmbeddingEuclideanDistance
from eager import Eager
from experiment import Experiments, Experiment
from matching.pair_to_vec import PairToVec
from prepared_models import classifier_factories
from run_configs import configs, EmbeddingInfo
from similarity_measures import (
    Levenshtein,
    GeneralizedJaccard,
    TriGram,
    EmbeddingConcatenation,
    NumberSimilarity,
    BertConcatenation,
    BertFeatureSimilarity,
    BertCosineSimilarity, bert_embed,
)
from write_results import find_existing_result_folder, write_result_files


def main():
    for dataset, emb_info in configs():
        run_for_dataset(dataset, emb_info)


def run_for_dataset(dataset, emb_info):
    import tensorflow as tf

    tf.reset_default_graph()

    rnd = random.Random(42)
    dataset.add_negative_samples(rnd)
    print(f"using {emb_info.name} on {dataset.name()}")
    existing_folder = find_existing_result_folder(emb_info.model)
    embeddings = get_embeddings(dataset, emb_info)

    all_pairs = (
        dataset.labelled_train_pairs
        + dataset.labelled_val_pairs
        + dataset.labelled_test_pairs
    )
    output_folder = existing_folder or emb_info.model.out_folder[:-1]

    kgs = dataset.kgs()
    cartesian_attr_combination = CartesianCombination(
        kgs,
        [NumberSimilarity()],
        [DateDistance()],
        [Levenshtein(), GeneralizedJaccard(), TriGram()],
    )
    no_attribute_combinations = CartesianCombination(kgs, [], [], [])
    all_to_one_concat = AllToOneCombination(
        kgs, [BertConcatenation(), BertCosineSimilarity()]
    )
    all_to_one_diff = AllToOneCombination(
        kgs, [BertFeatureSimilarity(), BertCosineSimilarity()]
    )

    embedding_measures = [EmbeddingEuclideanDistance(), EmbeddingConcatenation()]
    pair_to_vecs = [
        # lambda: PairToVec(
        #     embeddings,
        #     kgs,
        #     "SimAndEmb",
        #     cartesian_attr_combination,
        #     embedding_measures,
        # ),
        # lambda: PairToVec(
        #     embeddings,
        #     kgs,
        #     "OnlyEmb",
        #     no_attribute_combinations,
        #     [EmbeddingConcatenation()],
        # ),
        # lambda: PairToVec(embeddings, kgs, "OnlySim", cartesian_attr_combination, []),
        # lambda: PairToVec(
        #     embeddings, kgs, "AllConcatAndEmb", all_to_one_concat, embedding_measures
        # ),
        lambda: PairToVec(embeddings, kgs, "OnlyAllConcat", all_to_one_concat, []),
        lambda: PairToVec(
            embeddings, kgs, "AllDiffAndEmb", all_to_one_diff, embedding_measures
        ),
        lambda: PairToVec(embeddings, kgs, "OnlyAllDiff", all_to_one_diff, []),
    ]
    results_list = []
    for pvp in pair_to_vecs:
        pvp = pvp()
        print("loading pvp data")
        # loaded_pvp = PairToVec.load(embeddings, kgs, output_folder, pvp.name)
        # if loaded_pvp is None:
        print("no existing pvp data found, preparing similarity dataframe")
        pvp.prepare(all_pairs)
        pvp.save(output_folder)
        # else:
        #     pvp.set_prepared(loaded_pvp.all_sims, loaded_pvp.min_max, loaded_pvp.cols)
        #     print("loaded existed pvp data")

    # pair_to_vecs = [
    #     PairToVec.load(embeddings, kgs, output_folder, pvp.name) for pvp in pair_to_vecs
    # ]
    # kg1_dict = kgs.kg1.av_dict
    # kg2_dict = kgs.kg2.av_dict
    # e1_attrs = [v for v in kg1_dict.values()]
    # e2_attrs = [v for v in kg2_dict.values()]
    # v1s = [" ".join(_remove_type(v) for _, v in e1_attr) for e1_attr in e1_attrs]
    # v2s = [" ".join(_remove_type(v) for _, v in e2_attr) for e2_attr in e2_attrs]
    # bert_model = SentenceTransformer("distilbert-multilingual-nli-stsb-quora-ranking")
    # embeds_1 = [bert_embed(bert_model, v) for v in v1s]
    # embeds_2 = [bert_embed(bert_model, v) for v in v2s]
    # import sys
    # import json
    # sys.path.append(path.abspath('../corner/src'))
    # from corner import knn
    # neighbors_file = "bert_neighbors_minkowski_csls.json"
    # if not path.exists(neighbors_file) or True:
    #     # neighbors = get_nearest_neighbors.knn(embeddings[::2], embeddings[1::2], 50, metric="minkowski", hubness=None)
    #     neighbors = knn(embeds_1, embeds_2, 50, metric="minkowski", hubness="csls")
    #     neighbor_pairs = [(2 * int(i), 2 * int(j) + 1) for i, arr in enumerate(neighbors[1]) for j in arr]
    #     print(f"number of neighbor pairs: {len(neighbor_pairs)}")
    #     with open(neighbors_file, "w") as f:
    #         json.dump(neighbor_pairs, f)
    # else:
    #     with open(neighbors_file) as f:
    #         neighbor_pairs = json.load(f)
    #
    # eager_name, eager_classifier = classifier_factories[0]
    # eager_classifier = eager_classifier()
    # file_name = f"{output_folder}/{dataset.name().replace('/', '-')}-{emb_info.name}_{eager_name}_{pair_to_vecs[0].name}_bert_minkowski_csls_knn_pred.json"
    # eager = Eager(eager_classifier, pair_to_vecs[0], eager_name)
    # eager.fit(dataset.labelled_train_pairs, dataset.labelled_val_pairs)
    # predictions = eager.predict(neighbor_pairs)
    # with open(file_name, "w") as f:
    #     json.dump([(n[0], n[1], float(p)) for n, p in zip(neighbor_pairs, predictions)], f)
    #
    # with open(file_name) as f:
    #     predictions = json.load(f)
    # predictions = [tuple(l) for l in predictions]
    # print({e[2] for e in predictions})
    # import itertools as it
    # counts = {k: len([_ for _ in v if _[2] > 0.5]) for k, v in it.groupby(predictions, lambda e: e[0])}
    # print({k: v for k, v in counts.items() if v > 0})
    # print("finished writing predictions")
    # result = eager._eval.evaluate(dataset.labelled_test_pairs, predictions)
    # print(result)

        experiments = Experiments(
            output_folder,
            [
                Experiment(Eager(classifier_fac(), pvp, name))
                # for pair_to_vec in pair_to_vecs
                for name, classifier_fac in classifier_factories
            ],
            dataset,
        )

        results_list.extend(experiments.run())
    write_result_files(output_folder, dataset, emb_info.name, results_list)


def get_embeddings(dataset, emb_info: EmbeddingInfo):
    existing_embedding_folder = find_existing_embedding_folder(emb_info.model, dataset)
    if existing_embedding_folder is not None:
        embeddings = np.load(f"{existing_embedding_folder}/ent_embeds.npy")
    else:
        emb_info.model.set_kgs(dataset.kgs())
        emb_info.model.init()
        emb_info.model.run()
        emb_info.model.test()
        emb_info.model.save()
        embeddings = emb_info.extractor(emb_info.model)

    return embeddings


def find_existing_embedding_folder(embedding_model: BasicModel, dataset: Dataset):
    folder = f"data/Embeddings{dataset.data_size.value}K/{type(embedding_model).__name__}/{dataset.name()}"
    if not path.exists(folder):
        folder = embedding_model.out_folder
        folder = "/".join(folder.split("/")[:-2])
        dirs = sorted(
            [
                d
                for d in glob(f"{folder}/*")
                if embedding_model.args.dataset_division in d
            ]
        )
    else:
        dirs = sorted([d for d in glob(f"{folder}/*")])

    if any(dirs):
        return dirs[-1]
    return None


if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_start_method("spawn")
    main()
