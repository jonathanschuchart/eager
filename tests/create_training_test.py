import numpy as np
import pytest
from assertpy import assert_that
from src.similarity.create_training import (
    create_from_similarities,
    create_feature_vectors,
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
        (0, 12): {"Lev.0:0": 0.16, "euclidean": 2.34},
        (1, 2): {"Lev.2:2": 0.34, "euclidean": 3.4},
        (1, 3): {"Lev.2:2": 0.11, "euclidean": 8.4},
        (5, 6): {"euclidean": 5.3},
    }
    links = [(0, 12, 1), (1, 3, 0), (1, 2, 1)]

    features, labels, names, features_unlabeled = create_from_similarities(sims, links)

    assert ["Lev.0:0", "Lev.2:2", "euclidean"] == names
    np.testing.assert_array_equal(
        np.array([[0.16, np.nan, 2.34], [np.nan, 0.11, 8.4], [np.nan, 0.34, 3.4]]),
        features,
    )
    np.testing.assert_array_equal(np.array([1, 0, 1]), labels)
    assert_that(features_unlabeled).is_length(1)


def test_create_feature_vectors(loaded_kgs, embedding):
    links = [(0, 12, 1), (1, 3, 0), (1, 2, 1)]
    features, labels, feature_names, features_unlabeled = create_feature_vectors(
        embedding, links, loaded_kgs, 5
    )
    assert_that(features).is_length(len(links))
    assert_that(labels).is_length(len(links))
