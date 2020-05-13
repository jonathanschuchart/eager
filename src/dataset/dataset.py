import random
from typing import List, Tuple, Iterable, Union


def _get_positive_pairs(base_path, entity_type, source_names) -> List[Tuple[str, str]]:
    all_pairs = []
    with open(f"{base_path}/Curated/{entity_type}") as f:
        for line in f.readlines():
            entries = line.split(",")
            entries = [
                e.strip()
                for s in source_names
                for e in entries
                if e[45:49].lower() == s
            ]
            pairs = list(zip(entries[:-1], entries[1:]))
            all_pairs.extend(pairs)
    return all_pairs


def _split(
    train_size: float,
    val_size: float,
    entities: Iterable[str],
    pairs: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    rnd = random.Random()
    entities = _shuffle(entities, rnd)
    train_entities = set()
    train_pairs = set()
    val_entities = set()
    val_pairs = set()
    test_entities = set()
    test_pairs = set()
    for entity in entities:
        if len(train_pairs) < train_size * len(pairs):
            train_entities.add(entity)
            new_pairs = [
                pair
                for pair in pairs
                if (pair[0] == entity or pair[1] == entity)
                and pair[0] in train_entities
                and pair[1] in train_entities
            ]
            train_pairs.update(new_pairs)
        elif len(val_pairs) < val_size * len(pairs):
            val_entities.add(entity)
            new_pairs = [
                pair
                for pair in pairs
                if (pair[0] == entity or pair[1] == entity)
                and (pair[0] in val_entities or pair[0] in train_entities)
                and (pair[1] in val_entities or pair[1] in train_entities)
            ]
            val_pairs.update(new_pairs)
        else:
            test_entities.add(entity)
            new_pairs = [
                pair for pair in pairs if (pair[0] == entity or pair[1] == entity)
            ]
            test_pairs.update(new_pairs)
    return list(train_pairs), list(val_pairs), list(test_pairs)


def _shuffle(elems, rnd):
    elems = elems[:]
    rnd.shuffle(elems)
    return elems


class Dataset:
    all_pairs: List[Tuple[str, str]]
    train_pairs: List[Tuple[str, str]]
    val_pairs: List[Tuple[str, str]]
    test_pairs: List[Tuple[str, str]]

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        val_pairs: Union[List[Tuple[str, str]], float] = 0.2,
        test_pairs: Union[List[Tuple[str, str]], float] = 0.1,
    ):
        self.all_pairs = (
            pairs
            + (val_pairs if isinstance(val_pairs, list) else [])
            + (test_pairs if isinstance(test_pairs, list) else [])
        )
        self.train_pairs = pairs
        self.val_pairs = None
        self.test_pairs = None
        if isinstance(val_pairs, list):
            self.val_pairs = val_pairs
        if isinstance(test_pairs, list):
            self.test_pairs = test_pairs
        if type(val_pairs) == float or type(test_pairs) == float:
            val_size = val_pairs if type(val_pairs) == float else 0.0
            test_size = test_pairs if type(test_pairs) == float else 0.0
            train_size = 1 - val_size - test_size
            rnd = random.Random()
            _shuffle(self.train_pairs, rnd)
            entities = list({e for es in self.all_pairs for e in es})
            self.train_pairs, val, test = _split(
                train_size, val_size, entities, self.all_pairs
            )
            self.val_pairs = self.val_pairs or (
                val + test if isinstance(test_pairs, list) else val
            )
            self.test_pairs = self.test_pairs or (
                val + test if isinstance(val_pairs, list) else test
            )
