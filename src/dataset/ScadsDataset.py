import random
from typing import Dict, List, Tuple


def _split(
    train_size: float,
    val_size: float,
    entities: List[str],
    pairs: List[Tuple[str, str, int]],
) -> Tuple[
    List[Tuple[str, str, int]], List[Tuple[str, str, int]], List[Tuple[str, str, int]]
]:
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


class ScadsDataset:
    def __init__(
        self,
        base_path: str,
        sources: List[str],
        e_type_list,
        sample_internal_rate=0.5,
        sample_other_rate=0.5,
        train_size=0.7,
        val_size=0.1,
    ):
        sources = [s.lower() for s in sources]
        rnd = random.Random()
        self._all_pos_pairs = {
            e_type: _shuffle(
                ScadsDataset._get_positive_pairs(base_path, e_type, sources),
                rnd,
            )
            for e_type in e_type_list
        }
        self._entities_per_source_and_type = {
            source: {
                e_type: list({e for es in pairs for e in es if e[45:49].lower() == source})
                for e_type, pairs in self._all_pos_pairs.items()
            }
            for source in sources
        }
        all_entities = [
            e
            for entities_per_type in self._entities_per_source_and_type.values()
            for es in entities_per_type.values()
            for e in es
        ]
        self._entities_per_source = {
            source: [e for es in entities_per_type.values() for e in es]
            for source, entities_per_type in self._entities_per_source_and_type.items()
        }
        internal_neg_pairs = self._sample_internal_negative_pairs(sample_internal_rate)
        external_neg_pairs = self._sample_external_negative_pairs(sample_other_rate)
        all_pairs = (
            [(e0, e1, 1) for es in self._all_pos_pairs.values() for e0, e1 in es]
            + internal_neg_pairs
            + external_neg_pairs
        )
        self.train_pairs, self.val_pairs, self.test_pairs = _split(
            train_size, val_size, all_entities, all_pairs
        )

    def all_entities(self) -> Dict[str, List[str]]:
        return self._entities_per_source

    def _sample_internal_negative_pairs(
        self, sample_internal_rate
    ) -> List[Tuple[str, str, int]]:
        negative_pairs = []
        rnd = random.Random()
        sources = list(self._entities_per_source_and_type.keys())
        for e_type, pos_pairs in self._all_pos_pairs.items():
            type_neg_pairs = set()
            pos_pairs = set(pos_pairs)
            while len(type_neg_pairs) < len(pos_pairs) * sample_internal_rate:
                d1 = rnd.choice(sources)
                d2 = rnd.choice([d for d in sources if d != d1])
                e1 = rnd.choice(self._entities_per_source_and_type[d1][e_type])
                e2 = rnd.choice(self._entities_per_source_and_type[d2][e_type])
                if (
                    (e1, e2) not in pos_pairs
                    and (e2, e1) not in pos_pairs
                    and (e1, e2) not in type_neg_pairs
                    and (e2, e1) not in type_neg_pairs
                ):
                    type_neg_pairs.add((e1, e2))
            negative_pairs.extend(type_neg_pairs)
        return [(e0, e1, 0) for e0, e1 in negative_pairs]

    def _sample_external_negative_pairs(
        self, sample_external_rate
    ) -> List[Tuple[str, str, int]]:
        negative_pairs = set()
        rnd = random.Random()
        sources = list(self._entities_per_source_and_type.keys())
        types = list(self._all_pos_pairs.keys())
        all_pos_pair_num = sum(len(pairs) for pairs in self._all_pos_pairs.values())
        while len(negative_pairs) < all_pos_pair_num * sample_external_rate:
            d1 = rnd.choice(sources)
            d2 = rnd.choice([d for d in sources if d != d1])
            t1 = rnd.choice(types)
            t2 = rnd.choice([t for t in types if t != t1])
            e1 = rnd.choice(self._entities_per_source_and_type[d1][t1])
            e2 = rnd.choice(self._entities_per_source_and_type[d2][t2])
            if (e1, e2) not in negative_pairs and (e2, e1) not in negative_pairs:
                negative_pairs.add((e1, e2))
        return [(e0, e1, 0) for e0, e1 in negative_pairs]

    @staticmethod
    def _get_entities(base_path, entity_type, source_name):
        entities = []
        with open(f"{base_path}/Curated/{entity_type}") as f:
            for line in f.readlines():
                entries = line.split(",")
                for e in entries:
                    if e[45:49].lower() == source_name:
                        entities.append(e)
        return entities

    @staticmethod
    def _get_positive_pairs(
        base_path, entity_type, source_names
    ) -> List[Tuple[str, str]]:
        all_pairs = []
        with open(f"{base_path}/Curated/{entity_type}") as f:
            for line in f.readlines():
                entries = line.split(",")
                entries = [e for e in entries if e[45:49].lower() in source_names]
                pairs = list(zip(entries[:-1], entries[1:]))
                all_pairs.extend(pairs)
        return all_pairs
