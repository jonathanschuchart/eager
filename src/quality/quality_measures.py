from typing import Set, Tuple


def get_confusion_matrix(
    gold_pos: Set[Tuple[str, str]],
    gold_neg: Set[Tuple[str, str]],
    predictions: Set[Tuple[str, str]],
) -> Tuple[int, int, int]:
    """
        Calculates necessary confusion matrix based on partial gold standard
    :param gold_pos: tuple of positive gold tuples
    :param gold_neg: tuple of negative gold tuples
    :param predictions: prediction on partial gs
    :return: false_positive, true_positive, false_negative
    """
    true_positive = len(gold_pos & predictions)
    false_positive = len(predictions & gold_neg)
    false_negative = len(gold_pos - predictions)
    return false_positive, true_positive, false_negative


def precision(confusion_matrix: Tuple[int, int, int]) -> float:
    if confusion_matrix[1] + confusion_matrix[0] == 0:
        return 1
    return confusion_matrix[1] / (confusion_matrix[1] + confusion_matrix[0])


def recall(confusion_matrix: Tuple[int, int, int]) -> float:
    if (confusion_matrix[1] + confusion_matrix[2]) == 0:
        return 0
    return confusion_matrix[1] / (confusion_matrix[1] + confusion_matrix[2])


def fmeasure(confusion_matrix: Tuple[int, int, int]) -> float:
    p = precision(confusion_matrix)
    r = recall(confusion_matrix)
    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0


def canonical_tuples(duple: Tuple[str, str]) -> Tuple[str, str]:
    """ Returns tuple in lexicogrpahical order """
    if duple[0] < duple[1]:
        return duple
    else:
        return duple[1], duple[0]


def get_tuples_from_file_pos_neg(
    file_path: str,
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    tuples_pos = set()
    tuples_neg = set()
    with open(file_path) as input_file:
        for line in input_file:
            if line.startswith("left_spec"):
                continue
            duple = line.strip().split(",")
            ordered_tuple = canonical_tuples(duple)
            if len(duple) == 3:
                label = int(duple[2])
                if label == 1:
                    tuples_pos.add((ordered_tuple[0], ordered_tuple[1]))
                else:
                    tuples_neg.add((ordered_tuple[0], ordered_tuple[1]))
            else:
                tuples_pos.add((ordered_tuple[0], ordered_tuple[1]))
    return tuples_pos, tuples_neg


def get_quality_measures(pred_path, gold_path) -> Tuple[float, float, float]:
    """
    :param pred_path: path of prediction pairs file
    :param gold_path:  path of gold file
    :return: precision, recall, fmeasure
    """
    predictions_pos, _ = get_tuples_from_file_pos_neg(pred_path)
    gold_pos, gold_neg = get_tuples_from_file_pos_neg(gold_path)
    confusion_matrix = get_confusion_matrix(gold_pos, gold_neg, predictions_pos)
    return (
        precision(confusion_matrix),
        recall(confusion_matrix),
        fmeasure(confusion_matrix),
    )
