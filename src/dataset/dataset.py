from typing import List

import rdflib
import pandas as pd
from rdf2vec.converters import rdflib_to_kg
from rdf2vec.graph import KnowledgeGraph


class Dataset:
    kg: KnowledgeGraph
    train_entities: List[rdflib.URIRef]
    train_labels: pd.Series
    test_entities: List[rdflib.URIRef]
    test_labels: pd.Series
    all_entities: List[rdflib.URIRef]
    all_labels: List[float]

    def load_train_test(self, kg_file, train_file, test_file):
        # Define the label predicates, all triples with these predicates
        # will be excluded from the graph
        label_predicates = ["http://dl-learner.org/carcinogenesis#isMutagenic"]

        # Convert the rdflib to our KnowledgeGraph object
        self.kg = rdflib_to_kg(kg_file, label_predicates=label_predicates)

        test_data = pd.read_csv(test_file, sep="\t")
        train_data = pd.read_csv(train_file, sep="\t")

        self.train_entities = [rdflib.URIRef(x) for x in train_data["bond"]]
        self.train_labels = train_data["label_mutagenic"]

        self.test_entities = [rdflib.URIRef(x) for x in test_data["bond"]]
        self.test_labels = test_data["label_mutagenic"]

        self.all_entities = self.train_entities + self.test_entities
        self.all_labels = list(self.train_labels) + list(self.test_labels)
