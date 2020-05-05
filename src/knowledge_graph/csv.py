from typing import Dict, Any

import pandas as pd
from rdflib import Graph, URIRef, Literal
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from rdflib.graph import BNode
from rdflib.namespace import Namespace, RDF, XSD
from rdflib.term import Node
import urllib3
import networkx as nx
import matplotlib.pyplot as plt

schema = Namespace("http://schema.org/")


def convert_products_to_rdf(csv_file: str, name_attr: str = "name") -> Graph:
    df = pd.read_csv(csv_file, encoding="iso8859", keep_default_na=False)
    graph = Graph()
    graph.bind("schema", schema)
    brands = {}
    for _, row in df.iterrows():
        product = URIRef(str(row["id"]))
        offer = BNode()
        price = row["price"]
        graph.add((product, RDF.type, schema.Product))
        graph.add((product, schema.name, Literal(row[name_attr], datatype=XSD.string)))
        graph.add(
            (
                product,
                schema.description,
                Literal(row["description"], datatype=XSD.string),
            )
        )
        if "manufacturer" in row and row["manufacturer"] != "":
            graph.add(
                (
                    product,
                    schema.brand,
                    _get_brand_node(graph, row["manufacturer"], brands),
                )
            )
        if price != "":
            graph.add((product, schema.offers, offer))
            graph.add((offer, RDF.type, schema.Offer))
            graph.add((offer, schema.priceSpecification, _get_price_node(graph, price)))
    return graph


def _get_brand_node(graph: Graph, brand_name: str, brands: Dict[str, Node]) -> Node:
    if brand_name not in brands:
        brand_node = URIRef(urllib3.util.url.quote(brand_name))
        graph.add((brand_node, RDF.type, schema.Brand))
        graph.add((brand_node, schema.name, Literal(brand_name, datatype=XSD.string)))
        brands[brand_name] = brand_node
    return brands[brand_name]


def _get_price_node(graph: Graph, price: Any) -> Node:
    price_node = BNode()
    graph.add((price_node, RDF.type, schema.PriceSpecification))
    graph.add((price_node, schema.price, Literal(price, datatype=XSD.float)))
    graph.add((price_node, schema.priceCurrency, Literal("USD", datatype=XSD.string)))
    return price_node
