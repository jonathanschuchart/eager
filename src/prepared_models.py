from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from matching.classifiers import SkLearnMatcher


classifier_factories = [
    ("ada boost", AdaBoostClassifier),
    ("random forest 500", lambda: RandomForestClassifier(500)),
    ("gaussian naive bayes", GaussianNB),
    (
        "MLP",
        lambda: MLPClassifier(
            solver="adam", alpha=1e-5, hidden_layer_sizes=(200, 20), max_iter=500
        ),
    ),
]


def model_factories(pair_to_vec):
    return [
        lambda: SkLearnMatcher(pair_to_vec, AdaBoostClassifier(), "ada boost"),
        lambda: SkLearnMatcher(
            pair_to_vec, RandomForestClassifier(500), "random forest 500"
        ),
        lambda: SkLearnMatcher(pair_to_vec, GaussianNB(), "gaussian naive bayes"),
        lambda: SkLearnMatcher(
            pair_to_vec,
            MLPClassifier(
                solver="adam", alpha=1e-5, hidden_layer_sizes=(200, 20), max_iter=500,
            ),
            "MLP",
        ),
    ]
