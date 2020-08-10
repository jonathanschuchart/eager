import json
import os
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Iterable, Tuple
from tqdm import tqdm

sparql_help = {
    "dbpedia": {
        "endpoint": "http://dbpedia.org/sparql",
        "default_graph": "http://dbpedia.org",
    },
    # "yago": {
    #     "endpoint": "https://yago-knowledge.org/sparql/query",
    #     "default_graph": "https://yago-knowledge.org",
    #     "additional_prefix": "PREFIX yago: <http://yago-knowledge.org/resource/>",
    #     "resource_prefix": "yago:",
    # },
    # "wikidata": {
    #     "endpoint": "https://query.wikidata.org/",
    #     "default_graph": "https://www.wikidata.org/",
    # },
}


# def get_entity_types(kg, typestr):
#     type_id = kg.relations_id_dict[typestr]
#     # TODO is there a better way than these iterations?
#     types_entities = dict()
#     entities_types = dict()
#     # dicts are uri -> id
#     ent_ids = {v: k for k, v in kg.entities_id_dict.items()}
#     for entity, tuples in kg.rt_dict.items():
#         for t in tuples:
#             if t[0] == type_id:
#                 if t[1] not in types_entities:
#                     types_entities[t[1]] = []
#                 types_entities[t[1]].append(entity)
#                 # entities_types[entity] = dict()
#                 # entities_types[entity]["type"] = t[1]
#                 # entities_types[entity]["type_uri"] = ent_ids[t[1]]
#                 # entities_types[entity]["uri"] = ent_ids[entity]
#                 uri = ent_ids[entity]
#                 entities_types[uri] = dict()
#                 entities_types[uri]["type"] = t[1]
#                 entities_types[uri]["type_uri"] = ent_ids[t[1]]
#                 entities_types[uri]["id"] = entity
#     return entities_types, types_entities


def get_type_sets(datasets_path: str, output_path: str):
    types = set()
    all_files = [
        os.path.join(subdir, file)
        for subdir, dirs, files in os.walk(datasets_path)
        for file in files
    ]
    for file in tqdm(all_files):
        with open(file) as fp:
            try:
                data = json.load(fp)
                types |= set([item for elem in data.values() for item in elem])
            except Exception as e:
                print(file)
                print(e)
    types_dbpedia = set()
    for t in types:
        if "http://dbpedia.org/ontology/" in t:
            types_dbpedia.add(t)
    with open(os.path.join(output_path, "all_types.txt"), "w") as filehandle:
        filehandle.writelines("%s\n" % place for place in types)
    with open(os.path.join(output_path, "dbpedia_types.txt"), "w") as filehandle:
        filehandle.writelines("%s\n" % place for place in types_dbpedia)


def _sparql_super_class(type: str):
    super = []
    sparql = SPARQLWrapper(sparql_help["dbpedia"]["endpoint"])
    sparql.addDefaultGraph(sparql_help["dbpedia"]["default_graph"])
    ressource_uri = "<" + type + ">"
    query = (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> select ?super where {"
        + ressource_uri
        + "rdfs:subClassOf ?super}"
    )
    sparql.setQuery(query)
    try:
        sparql.setReturnFormat(JSON)
        results = sparql.query()
        triples = results.convert()
        for t in triples["results"]["bindings"]:
            super.append(t["super"]["value"])
    except Exception:
        print("query failed")
        return None
    return super


def get_superclasses(set_path: str, out_path: str):
    super_types = dict()
    with open(set_path, "r") as set_file:
        for type in tqdm(set_file):
            type = type.strip()
            super = []
            next_it = []
            force_quit = 25
            query_type = type
            while "http://www.w3.org/2002/07/owl#Thing" not in next_it:
                next_it = _sparql_super_class(query_type)
                force_quit -= 1
                if next_it is None or len(next_it) == 0:
                    break
                for n in next_it:
                    if "http://dbpedia.org/ontology/" in n:
                        super.append(next_it)
                        if force_quit == 0 or query_type == n:
                            force_quit = 0
                            break
                        query_type = n
                if force_quit == 0:
                    break
            super_types[type] = super
    with open(os.path.join(out_path, "superclasses.json"), "w") as out_file:
        json.dump(super_types, out_file)


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
        print(query)
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
    if sys.argv[3] == "collect":
        get_all_types(input_path, output_path)
    elif sys.argv[3] == "set":
        get_type_sets(input_path, output_path)
    elif sys.argv[3] == "superclass":
        get_superclasses(input_path, output_path)
