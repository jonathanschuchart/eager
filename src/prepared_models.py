from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

classifiers = {
    "RF": lambda: RandomForestClassifier(500),
    "MLP": lambda: MLPClassifier(
        solver="adam", alpha=1e-5, hidden_layer_sizes=(200, 20), max_iter=500
    ),
}
