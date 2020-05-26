import numpy as np
import pytest
from assertpy import assert_that
from src.similarity.create_training import (
    create_labeled_similarity_frame,
    create_feature_similarity_frame,
)
from openea.modules.load.kgs import read_kgs_from_folder


@pytest.fixture
def loaded_kgs():
    return read_kgs_from_folder(
        "data/OpenEA/D_W_15K_V1/",
        "721_5fold/1/",
        "mapping",
        True,
        remove_unlinked=False,
    )


@pytest.fixture
def embedding():
    return np.load("tests/test_kgs/slice_ent_emb.npy")


def test_create_from_similarities():
    sims = {
        (0, 12): {"Lev.0:0": 0.2142857142857143, "euclidean": 2.548967831773},
        (1, 2): {"euclidean": 9.445619344428064},
        (1, 3): {"Lev.1:1": 0.13793103448275867, "euclidean": 8.4},
        (5, 6): {"euclidean": 8.911974600748268},
    }
    links = [(0, 12, 1), (1, 3, 0), (1, 2, 0)]

    sim_frame = create_labeled_similarity_frame(sims, links)

    assert_that(sim_frame.loc["0,12"]["euclidean"]).is_close_to(0.77, 0.5)
    assert_that(sim_frame.loc["0,12"]["Lev.0:0"]).is_close_to(
        sims[(0, 12)]["Lev.0:0"], 0.05
    )
    assert_that(sim_frame.loc["1,2"]["euclidean"]).is_close_to(0, 0.05)

    assert_that(sim_frame.loc["0,12"]["label"]).is_equal_to(1)
    assert_that(sim_frame.loc["1,3"]["label"]).is_equal_to(0)
    assert_that(sim_frame.loc["1,2"]["label"]).is_equal_to(0)


def test_create_feature_frame(loaded_kgs, embedding):
    links = [(0, 12, 1), (1, 3, 0), (1, 2, 0)]
    sim_frame = create_feature_similarity_frame(embedding, links, loaded_kgs, 5)
    assert_that(sim_frame.loc["0,12"]["euclidean"]).is_close_to(0.77, 0.05)
    assert_that(sim_frame.loc["0,12"]["Lev.0:0"]).is_close_to(0.21, 0.05)
    assert_that(sim_frame.loc["1,2"]["euclidean"]).is_close_to(0, 0.05)

    assert_that(sim_frame.loc["0,12"]["label"]).is_equal_to(1)
    assert_that(sim_frame.loc["1,3"]["label"]).is_equal_to(0)
    assert_that(sim_frame.loc["1,2"]["label"]).is_equal_to(0)
