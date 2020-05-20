import pickle
import pytest
import numpy as np
from assertpy import assert_that
from src.similarity.similarities import (
    calculate_from_embeddings,
    calculate_from_embeddings_with_training,
    align_attributes,
)
from openea.modules.load.kgs import KGs, read_kgs_from_folder


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


def test_align_attrs():
    e1_attrs = {
        1: "123",
        3: '"123"^^<http://www.w3.org/2001/XMLSchema#double>',
        5: "test",
    }
    e2_attrs = {
        1: "123",
        4: '"123"^^<http://www.w3.org/2001/XMLSchema#double>',
        5: "other",
    }
    aligned = align_attributes(e1_attrs, e2_attrs, False)
    assert_that(aligned).contains((1, 1), (5, 5), (3, 4))

    trivial = align_attributes(e1_attrs, e2_attrs)
    assert_that(trivial).contains((1, 1), (5, 5))
    assert_that(trivial).does_not_contain((3, 4))


def test_align_attrs_complex():
    e1_attrs = {
        20: "Canadian ice hockey player",
        76: "87th Overall",
        36: '"1.9304"^^<http://www.w3.org/2001/XMLSchema#double>',
        6: '"1991-12-06"^^<http://www.w3.org/2001/XMLSchema#date>',
        46: '"98431.2"^^<http://www.w3.org/2001/XMLSchema#double>',
        56: '"2010"^^<http://www.w3.org/2001/XMLSchema#gYear>',
        10: '"2012"^^<http://www.w3.org/2001/XMLSchema#gYear>',
        8: '"1991"^^<http://www.w3.org/2001/XMLSchema#gYear>',
        0: "Melchiori, Julian",
        68: "Left",
    }
    e2_attrs = {
        151: "121474",
        1: "Canadian ice hockey player",
        11: '"1991-12-06"^^<http://www.w3.org/2001/XMLSchema#date>',
        157: "73778",
    }
    aligned = align_attributes(e1_attrs, e2_attrs, False)
    assert_that(aligned).contains(
        (20, 151),
        (20, 1),
        (20, 157),
        (76, 151),
        (76, 1),
        (76, 157),
        (0, 151),
        (0, 1),
        (0, 157),
        (68, 151),
        (68, 1),
        (68, 157),
        (6, 11),
    )

    trivial = align_attributes(e1_attrs, e2_attrs)
    assert_that(trivial).is_length(0)


def test_calculate_from_embeddings(loaded_kgs, embedding):
    similarities = calculate_from_embeddings(embedding, loaded_kgs, 5, "euclidean")
    assert_that(similarities).does_not_contain_key((0, 0))
    assert_that(similarities[(0, 12)]).contains_key(
        "Lev.30:30",
        "GenJac.30:30",
        "Trigram.30:30",
        "NumberDist.38:38",
        "NumberDist.40:40",
        "NumberDist.60:60",
        # "NumberDist.48:60",
        # "NumberDist.40:60",
        # "Lev.28:136",
        # "GenJac.28:136",
        # "Trigram.28:136",
        # "Lev.28:164",
        # "GenJac.28:164",
        # "Trigram.28:164",
        "Lev.0:0",
        "GenJac.0:0",
        "Trigram.0:0",
        "euclidean",
    )
    assert_that(similarities[(0, 12)]["Lev.0:0"]).is_greater_than(0.0)


def test_calculate_from_embeddings_with_training(loaded_kgs, embedding):
    similarities = calculate_from_embeddings_with_training(
        embedding, [(0, 12, 1)], loaded_kgs, "euclidean"
    )
    assert_that(similarities).does_not_contain_key((0, 0))
    assert_that(similarities[(0, 12)]).contains_key(
        "Lev.30:30",
        "GenJac.30:30",
        "Trigram.30:30",
        "NumberDist.38:38",
        "NumberDist.40:40",
        "NumberDist.60:60",
        # "NumberDist.48:60",
        # "NumberDist.40:60",
        # "Lev.28:136",
        # "GenJac.28:136",
        # "Trigram.28:136",
        # "Lev.28:164",
        # "GenJac.28:164",
        # "Trigram.28:164",
        "Lev.0:0",
        "GenJac.0:0",
        "Trigram.0:0",
        "euclidean",
    )
    assert_that(similarities[(0, 12)]["Lev.0:0"]).is_greater_than(0.0)
