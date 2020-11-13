from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


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
