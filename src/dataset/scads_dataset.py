import random
from typing import Dict, List, Tuple

import rdflib
from openea.models.basic_model import BasicModel
from openea.modules.load.kg import KG
from openea.modules.load.kgs import KGs
from rdflib import RDF

from dataset.dataset import Dataset
from knowledge_graph.rdf_to_openea import convert_rdf_to_openea


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


def _shuffle(elems, rnd):
    elems = elems[:]
    rnd.shuffle(elems)
    return elems


def read_kg(base_path: str, source: str, ids: Dict[str, int]) -> KG:
    g = rdflib.graph.Graph()
    g.parse(f"{base_path}/{source}_snippet.nt", format="nt")
    return convert_rdf_to_openea(g, ids)


class ScadsDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        source1: str,
        source2: str,
        model: BasicModel,
        train_size=0.7,
        val_size=0.2,
    ):
        source1 = source1.lower()
        source2 = source2.lower()
        e_type_list = ["company", "episode", "movie", "person", "tvSeries"]
        rnd = random.Random()
        ids = {}

        kg1 = read_kg(base_path, source1, ids)
        kg2 = read_kg(base_path, source2, ids)
        pos_pairs_per_type = {
            e_type: _shuffle(
                _get_positive_pairs(base_path, e_type, [source1, source2]), rnd,
            )
            for e_type in e_type_list
        }

        all_pairs = [
            # (ids[e0], ids[e1], 1)
            (e0, e1)
            for es in pos_pairs_per_type.values()
            for e0, e1 in es
            if e0 in ids and e1 in ids
        ]
        super().__init__(
            kg1, kg2, all_pairs, val_ratio=val_size, test_ratio=train_size - val_size
        )
        self._kgs = KGs(
            kg1,
            kg2,
            self.labelled_train_pairs,
            self.labelled_test_pairs,
            self.labelled_val_pairs,
            mode=model.args.alignment_module,
            ordered=model.args.ordered,
        )
        self.labelled_train_pairs = [(e[0], e[1], 1) for e in self._kgs.train_links]
        self.labelled_test_pairs = [(e[0], e[1], 1) for e in self._kgs.test_links]
        self.labelled_val_pairs = [(e[0], e[1], 1) for e in self._kgs.valid_links]

        self._name = "ScaDS_" + source1 + "_" + source2

    def name(self):
        return self._name

    def kgs(self):
        return self._kgs
