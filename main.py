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
    BertFeatureSimilarity,
    BertCosineSimilarity,
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
        kgs, [Levenshtein(), GeneralizedJaccard(), TriGram()]
    )
    all_to_one_diff = AllToOneCombination(
        kgs, [BertFeatureSimilarity(), BertCosineSimilarity()]
    )

    embedding_measures = [EmbeddingEuclideanDistance(), EmbeddingConcatenation()]
    support_threshold = 0.1
    pair_to_vecs = [
        # lambda: PairToVec(
        #     embeddings,
        #     kgs,
        #     "SimAndEmb",
        #     cartesian_attr_combination,
        #     embedding_measures,
        #     support_threshold,
        # ),
        # lambda: PairToVec(
        #     embeddings,
        #     kgs,
        #     "OnlyEmb",
        #     no_attribute_combinations,
        #     [EmbeddingConcatenation()],
        #     support_threshold,
        # ),
        # lambda: PairToVec(
        #     embeddings,
        #     kgs,
        #     "OnlySim",
        #     cartesian_attr_combination,
        #     [],
        #     support_threshold,
        # ),
        lambda: PairToVec(
            embeddings, kgs, "AllConcatAndEmb", all_to_one_concat, embedding_measures
        ),
        lambda: PairToVec(embeddings, kgs, "OnlyAllConcat", all_to_one_concat, []),
        # lambda: PairToVec(
        #     embeddings, kgs, "AllDiffAndEmb", all_to_one_diff, embedding_measures
        # ),
        # lambda: PairToVec(embeddings, kgs, "OnlyAllDiff", all_to_one_diff, []),
    ]
    results_list = []
    for pvp in pair_to_vecs:
        pvp = pvp()
        pvp.prepare(dataset.labelled_train_pairs)
        experiments = Experiments(
            output_folder,
            [
                Experiment(Eager(classifier_fac(), pvp, name + " knn"))
                # for pair_to_vec in pair_to_vecs
                for name, classifier_fac in classifier_factories
            ],
            dataset,
            None
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
