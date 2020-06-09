from enum import Enum
from urllib import parse

from dataset.dataset import Dataset
from knowledge_graph import from_csv
import pandas as pd

from knowledge_graph.rdf_to_openea import convert_rdf_to_openea


class CsvType(Enum):
    products = 1
    articles = 2


class CsvDataset(Dataset):
    def __init__(self, type: CsvType, csv1: str, csv2: str, links_csv: str):
        ids = {}
        if type == CsvType.products:
            kg1 = convert_rdf_to_openea(from_csv.convert_products_to_rdf(csv1), ids)
            kg2 = convert_rdf_to_openea(from_csv.convert_products_to_rdf(csv2), ids)
        else:
            kg1 = convert_rdf_to_openea(from_csv.convert_articles_to_rdf(csv1), ids)
            kg2 = convert_rdf_to_openea(from_csv.convert_articles_to_rdf(csv2), ids)
        links = pd.read_csv(links_csv)
        links = [
            (ids[parse.quote(str(x[0]))], ids[parse.quote(str(x[1]))], 1)
            for x in links.to_numpy()
        ]
        super().__init__(kg1, kg2, links)
        self._name = csv1.split("/")[1]

    def name(self):
        return self._name
