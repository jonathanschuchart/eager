from typing import List, Any, Union, Dict
import numpy as np
from rdf2vec import RandomWalker, RDF2VecTransformer
from rdf2vec.graph import KnowledgeGraph

from src.embedding.embedding import Embedding


class Rdf2VecConfig:
    def __init__(
        self,
        embedding_size=500,
        sg=1,
        walk_depth=4,
        walk_num=float("inf"),
        max_iter=200,
    ):
        self.embedding_size = embedding_size
        self.sg = sg
        self.max_iter = max_iter
        self.walk_depth = walk_depth
        self.walk_num = walk_num


class Rdf2Vec(Embedding):
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        rdf2vec_config: Union[Dict[str, Any], Rdf2VecConfig],
    ):
        super().__init__()
        if type(rdf2vec_config) == dict:
            conf = Rdf2VecConfig()
            conf.__dict__.update(rdf2vec_config)
        else:
            conf = rdf2vec_config
        self._kg = knowledge_graph
        random_walker = RandomWalker(conf.walk_depth, conf.walk_num)
        self._transformer = RDF2VecTransformer(
            vector_size=conf.embedding_size,
            walkers=[random_walker],
            sg=conf.sg,
            max_iter=500,
        )

    def fit(self, entities: List[Any]):
        self._transformer.fit(self._kg, entities)

    def embed(self, entities: List[Any]) -> List[np.array]:
        return self._transformer.transform(self._kg, entities)
