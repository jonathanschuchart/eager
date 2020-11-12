import numpy as np
import pytest
from assertpy import assert_that
import os

from attribute_features import CartesianCombination
from distance_measures import EmbeddingEuclideanDistance, DateDistance
from matching.pair_to_vec import PairToVec
from similarity_measures import NumberSimilarity, Levenshtein
from src.similarity.create_training import (
    create_labeled_similarity_frame,
    create_feature_similarity_frame,
)
from openea.modules.load.kgs import read_kgs_from_folder


@pytest.fixture
def loaded_kgs():
    path = "data/OpenEA/D_W_15K_V1/"
    if not os.path.exists(path):
        path = os.path.join("..", path)
    return read_kgs_from_folder(
        path, "721_5fold/1/", "mapping", True, remove_unlinked=False,
    )


@pytest.fixture
def embedding():
    return np.load("tests/test_kgs/slice_ent_emb.npy")


def test_create_from_similarities(embedding, loaded_kgs):
    sims = {
        (0, 12): {
            "Lev.0:0": 0.2142857142857143,
            "euclidean": 2.548967831773,
            "label": 1,
        },
        (1, 2): {"euclidean": 9.445619344428064, "label": 0},
        (1, 3): {"Lev.1:1": 0.13793103448275867, "euclidean": 8.4, "label": 0},
        (5, 6): {"euclidean": 8.911974600748268},
    }
    links = [(0, 12, 1), (1, 3, 0), (1, 2, 0)]

    pvp = PairToVec(embedding, loaded_kgs, "some_name", None, [])

    sim_frame, _, _ = pvp._create_labeled_similarity_frame(sims, ["euclidean"])

    assert_that(sim_frame.loc[0, 12]["euclidean"]).is_close_to(0.77, 0.5)
    assert_that(sim_frame.loc[0, 12]["Lev.0:0"]).is_close_to(
        sims[(0, 12)]["Lev.0:0"], 0.05
    )
    assert_that(sim_frame.loc[1, 2]["euclidean"]).is_close_to(0, 0.05)

    assert_that(sim_frame.loc[0, 12]["label"]).is_equal_to(1)
    assert_that(sim_frame.loc[1, 3]["label"]).is_equal_to(0)
    assert_that(sim_frame.loc[1, 2]["label"]).is_equal_to(0)


def test_create_feature_frame(loaded_kgs, embedding):
    links = [(0, 12, 1), (1, 3, 0), (1, 2, 0)]
    alignment = CartesianCombination(loaded_kgs, [], [DateDistance()], [Levenshtein()])
    pvp = PairToVec(
        embedding, loaded_kgs, "some_name", alignment, [EmbeddingEuclideanDistance()]
    )
    pvp.prepare(links)
    sim_frame = pvp.all_sims
    assert_that(sim_frame.loc[1, 3]["EmbeddingEuclideanDistance"]).is_close_to(
        0.08, 0.05
    )
    assert_that(sim_frame.loc[0, 12]["Levenshtein.0:0"]).is_close_to(0.21, 0.05)
    assert_that(sim_frame.loc[1, 2]["EmbeddingEuclideanDistance"]).is_close_to(0, 0.05)
