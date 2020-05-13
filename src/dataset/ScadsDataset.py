import random
from typing import Dict, List, Tuple

from dataset.dataset import Dataset


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
    entities: List[str],
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


class ScadsDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        sources: List[str],
        e_type_list,
        train_size=0.7,
        val_size=0.2,
    ):
        sources = [s.lower() for s in sources]
        rnd = random.Random()
        self._all_pos_pairs = {
            e_type: _shuffle(_get_positive_pairs(base_path, e_type, sources), rnd,)
            for e_type in e_type_list
        }
        self._entities_per_source_and_type = {
            source: {
                e_type: list(
                    {e for es in pairs for e in es if e[45:49].lower() == source}
                )
                for e_type, pairs in self._all_pos_pairs.items()
            }
            for source in sources
        }
        self._entities_per_source = {
            source: [e for es in entities_per_type.values() for e in es]
            for source, entities_per_type in self._entities_per_source_and_type.items()
        }
        all_pairs = [(e0, e1) for es in self._all_pos_pairs.values() for e0, e1 in es]
        super().__init__(all_pairs, train_size, val_size)
