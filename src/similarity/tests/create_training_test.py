import numpy as np
from similarity.create_training import create_from_similarities


def test_create_from_similarities():
    sims = {
        (0, 12): {"Lev.0:0": 0.16, "euclidean": 2.34},
        (1, 2): {"Lev.2:2": 0.34, "euclidean": 3.4},
        (1, 3): {"Lev.2:2": 0.11, "euclidean": 8.4},
        (5, 6): {"euclidean": 5.3},
    }
    links = [(0, 12, 1), (1, 3, 0), (1, 2, 1)]

    features, labels, names = create_from_similarities(sims, links)

    assert ["Lev.0:0", "Lev.2:2", "euclidean"] == names
    np.testing.assert_array_equal(
        np.array([[0.16, np.nan, 2.34], [np.nan, 0.11, 8.4], [np.nan, 0.34, 3.4]]),
        features,
    )
    np.testing.assert_array_equal(np.array([1, 0, 1]), labels)
