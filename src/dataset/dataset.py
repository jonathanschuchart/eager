import enum
import os
import random
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple
from zipfile import ZipFile

import wget
from openea.modules.load.kg import KG
from openea.modules.load.kgs import KGs


class DataSize(enum.Enum):
    K15 = 15
    K100 = 100


def _split(
    train_size: float,
    val_size: float,
    entities: Iterable[int],
    pairs: List[Tuple[int, ...]],
    rnd: random.Random,
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], List[Tuple[int, ...]]]:
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


def sample_negative(
    pos_samples: List[Tuple[int, ...]], rnd: random.Random
) -> List[Tuple[int, int, int]]:
    negative_pairs = set()
    entities_left = list({e[0] for e in pos_samples})
    entities_right = list({e[1] for e in pos_samples})
    pos_set = set(pos_samples)
    while len(negative_pairs) < len(pos_samples):
        e1 = rnd.choice(entities_left)
        e2 = rnd.choice(entities_right)
        if (
            e1 != e2
            and (e1, e2, 1) not in pos_set
            and (e2, e1, 1) not in pos_set
            and (e1, e2) not in negative_pairs
            and (e2, e1) not in negative_pairs
        ):
            negative_pairs.add((e1, e2))
    return [(e0, e1, 0) for e0, e1 in negative_pairs]


class Dataset:
    labelled_train_pairs: List[Tuple[int, int, int]]
    labelled_val_pairs: List[Tuple[int, int, int]]
    labelled_test_pairs: List[Tuple[int, int, int]]
    kg1: KG
    kg2: KG
    data_size: DataSize

    def __init__(
        self,
        kg1: KG,
        kg2: KG,
        rnd: random.Random,
        labelled_pairs: List[Tuple[int, int, int]],
        labelled_val_pairs: List[Tuple[int, int, int]] = None,
        val_ratio: Optional[float] = 0.2,
        labelled_test_pairs: List[Tuple[int, int, int]] = None,
        test_ratio: Optional[float] = 0.1,
    ):
        self.kg1 = kg1
        self.kg2 = kg2
        all_labelled_pairs = (
            labelled_pairs
            + (labelled_val_pairs if labelled_val_pairs is not None else [])
            + (labelled_test_pairs if labelled_test_pairs is not None else [])
        )
        self.labelled_train_pairs = labelled_pairs
        self.labelled_val_pairs = labelled_val_pairs
        self.labelled_test_pairs = labelled_test_pairs
        if labelled_val_pairs is None or labelled_test_pairs is None:
            train_ratio = 1 - val_ratio - test_ratio
            _shuffle(self.labelled_train_pairs, rnd)
            entities = list({e for es in all_labelled_pairs for e in es})
            self.labelled_train_pairs, val, test = _split(
                train_ratio, val_ratio, entities, all_labelled_pairs, rnd
            )
            self.labelled_val_pairs = self.labelled_val_pairs or (
                val + test if isinstance(labelled_test_pairs, list) else val
            )
            self.labelled_test_pairs = self.labelled_test_pairs or (
                val + test if isinstance(labelled_val_pairs, list) else test
            )

    def _get_download_dir(self):
        if self._data_folder.endswith(os.sep):
            return os.path.split(os.path.split(self._data_folder)[0])[0]
        return os.path.split(self._data_folder)[0]

    def download_and_unzip(self):
        if not os.path.exists(self._data_folder):
            target_dir = self._get_download_dir()
            zip_file_path = target_dir + ".zip"
            print(f"Downloading {self.__class__.__name__} datasets to {zip_file_path}")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            wget.download(self.__class__._download_url, zip_file_path)
            with ZipFile(zip_file_path, "r") as zip_obj:
                tmp_dir = os.path.split(target_dir)[0]
                zip_obj.extractall(tmp_dir)
                os.rename(os.path.join(tmp_dir, "OpenEA_dataset_v1.1"), target_dir)
            os.remove(zip_file_path)

    def add_negative_samples(self, rnd: random.Random):
        neg_train_pairs = sample_negative(self.labelled_train_pairs, rnd)
        neg_val_pairs = sample_negative(self.labelled_val_pairs, rnd)
        neg_test_pairs = sample_negative(self.labelled_test_pairs, rnd)
        self.labelled_train_pairs += neg_train_pairs
        rnd.shuffle(self.labelled_train_pairs)
        self.labelled_val_pairs += neg_val_pairs
        rnd.shuffle(self.labelled_val_pairs)
        self.labelled_test_pairs += neg_test_pairs
        rnd.shuffle(self.labelled_test_pairs)

    def kgs(self):
        return KGs(
            self.kg1,
            self.kg2,
            [e[:2] for e in self.labelled_train_pairs if e[2] == 1],
            [e[:2] for e in self.labelled_test_pairs if e[2] == 1],
            [e[:2] for e in self.labelled_val_pairs if e[2] == 1],
            mode="mapping",
            ordered=False,
        )

    @abstractmethod
    def name(self) -> str:
        pass
