from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from matching.sklearn import SkLearnMatcher

model_factories = [
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, LogisticRegression(), "logistic regression"
    ),
    lambda pair_to_vec: SkLearnMatcher(pair_to_vec, AdaBoostClassifier(), "ada boost"),
    lambda pair_to_vec: SkLearnMatcher(pair_to_vec, svm.SVC(), "svc"),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(20), "random forest 20"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(50), "random forest 50"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(100), "random forest 100"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(200), "random forest 200"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, RandomForestClassifier(500), "random forest 500"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, DecisionTreeClassifier(), "decision tree"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec, GaussianNB(), "gaussian naive bayes"
    ),
    lambda pair_to_vec: SkLearnMatcher(
        pair_to_vec,
        MLPClassifier(
            solver="adam", alpha=1e-5, hidden_layer_sizes=(200, 20), max_iter=500,
        ),
        "MLP",
    ),
]
