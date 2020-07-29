import json
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Iterable, Tuple
from tqdm import tqdm

sparql_help = {
    "dbpedia": {
        "endpoint": "http://dbpedia.org/sparql",
        "default_graph": "http://dbpedia.org",
    },
    "yago": {
        "endpoint": "https://yago-knowledge.org/sparql/query",
        "default_graph": "https://yago-knowledge.org",
        "additional_prefix": "PREFIX yago: <http://yago-knowledge.org/resource/>",
        "resource_prefix": "yago:",
    },
    "wikidata": {
        "endpoint": "https://query.wikidata.org/",
        "default_graph": "https://www.wikidata.org/",
    },
}


def get_entity_types(kg, typestr):
    type_id = kg.relations_id_dict[typestr]
    # TODO is there a better way than these iterations?
    types_entities = dict()
    entities_types = dict()
    # dicts are uri -> id
    ent_ids = {v: k for k, v in kg.entities_id_dict.items()}
    for entity, tuples in kg.rt_dict.items():
        for t in tuples:
            if t[0] == type_id:
                if t[1] not in types_entities:
                    types_entities[t[1]] = []
                types_entities[t[1]].append(entity)
                # entities_types[entity] = dict()
                # entities_types[entity]["type"] = t[1]
                # entities_types[entity]["type_uri"] = ent_ids[t[1]]
                # entities_types[entity]["uri"] = ent_ids[entity]
                uri = ent_ids[entity]
                entities_types[uri] = dict()
                entities_types[uri]["type"] = t[1]
                entities_types[uri]["type_uri"] = ent_ids[t[1]]
                entities_types[uri]["id"] = entity
    return entities_types, types_entities


def get_ressource_types(ressource_uri: str, data_key=None):
    if data_key:
        return get_types_from_endpoint(sparql_help[data_key], ressource_uri)
    for k, query_info in sparql_help.items():
        if k in ressource_uri:
            return get_types_from_endpoint(query_info, ressource_uri)
    else:
        print(f"Unknown endpoint in ressource {ressource_uri}")


def get_types_from_endpoint(query_info: dict, ressource_uri: str):
    types = []
    sparql = SPARQLWrapper(query_info["endpoint"])
    sparql.addDefaultGraph(query_info["default_graph"])
    additional_prefix = (
        query_info["additional_prefix"] if "additional_prefix" in query_info else ""
    )
    ressource_uri = (
        query_info["resource_prefix"] + ressource_uri
        if "resource_prefix" in query_info
        else "<" + ressource_uri + ">"
    )
    query = (
        additional_prefix
        + "\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> select ?type where {"
        + ressource_uri
        + " rdf:type ?type } LIMIT 50"
    )
    sparql.setQuery(query)
    try:
        sparql.setReturnFormat(JSON)
        results = sparql.query()
        triples = results.convert()
        for t in triples["results"]["bindings"]:
            types.append(t["type"]["value"])
    except Exception:
        print("query failed")
    return types


def get_link_types(links: Iterable[Tuple[str, str]], data_keys=["dbpedia"]):
    typed_links = {}
    for link in tqdm(links):
        if len(data_keys) == 1:
            # dbpedia provides all necesary info
            types = get_ressource_types(link[0], data_key=data_keys[0])
            typed_links[link[0]] = types
            typed_links[link[1]] = types
        else:
            typed_links[link[0]] = get_ressource_types(link[0], data_key=data_keys[0])
            typed_links[link[1]] = get_ressource_types(link[1], data_key=data_keys[1])
    return typed_links


def get_all_types(link_file_path: str, out_path: str):
    with open(link_file_path, "r") as link_file:
        links = [line.strip().split("\t") for line in link_file]
        typed_links = get_link_types(links)
        with open(out_path, "w") as out_file:
            json.dump(typed_links, out_file)


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
