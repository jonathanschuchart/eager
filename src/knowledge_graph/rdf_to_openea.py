from typing import Dict

import rdflib
from openea.modules.load.kg import KG
from rdflib import Graph


def is_relation(s) -> bool:
    return isinstance(s, rdflib.term.URIRef) or isinstance(s, rdflib.term.BNode)


def convert_rdf_to_openea(g: Graph, ids: Dict[str, int]) -> KG:
    relation_triples = [
        (o.__str__(), p.__str__(), s.__str__())
        for o, p, s in g.triples((None, None, None))
        if is_relation(s)
    ]
    for triple in relation_triples:
        for o in [triple[0], triple[2]]:
            if o not in ids:
                ids[o] = len(ids)

    def attribute_str(v):
        value = v.__str__().replace("\n", " ").replace("\t", " ")
        if v.datatype.__str__() == "http://www.w3.org/2001/XMLSchema#string":
            return value
        else:
            return f'"{value}"^^<{v.datatype.__str__()}>'

    attribute_triples = [
        (o.__str__(), p.__str__(), attribute_str(s))
        for o, p, s in g.triples((None, None, None))
        if not is_relation(s)
    ]
    for triple in attribute_triples:
        if triple[0] not in ids:
            ids[triple[0]] = len(ids)

    return KG(relation_triples, attribute_triples)
