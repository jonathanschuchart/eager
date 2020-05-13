from typing import Dict, Any

import pandas as pd
from rdflib import Graph, URIRef, Literal
from rdflib.graph import BNode
from rdflib.namespace import Namespace, RDF, XSD
from rdflib.term import Node
from urllib import parse

schema = Namespace("http://schema.org/")


def convert_articles_to_rdf(csv_file: str, encoding: str = "iso8859") -> Graph:
    """
    Converts each article described in the csv as a subgraph in rdf:
                                     headline
                                     /
                        ScholarlyArticle -- Author(s)
                                |
                              Event
                            /     \
                        Place     date

    :param csv_file: The file name of the csv file to be used
    :param encoding: The character encoding of the file
    :return: The entire Knowledge Graph for this file
    """
    df = pd.read_csv(csv_file, encoding=encoding, keep_default_na=False, dtype=str)
    graph = Graph()
    graph.bind("schema", schema)
    all_authors = {}
    events = {}
    venues = {}
    for _, row in df.head(1).iterrows():
        article = URIRef(str(row["id"]))
        graph.add((article, RDF.type, schema.ScholarlyArticle))
        graph.add(
            (article, schema.headline, Literal(row["title"], datatype=XSD.string))
        )
        authors = row["authors"].split(", ")
        for author in authors:
            if author not in all_authors:
                author_node = URIRef(parse.quote(author))
                graph.add((author_node, RDF.type, schema.Person))
                graph.add(
                    (author_node, schema.name, Literal(author, datatype=XSD.string))
                )
                all_authors[author] = author_node
            graph.add((article, schema.author, all_authors[author]))
        venue = row["venue"]
        year = row["year"]
        event = f"{venue}-{year}"
        if event not in events:
            event_node = URIRef(parse.quote(event))
            graph.add((event_node, RDF.type, schema.Event))
            graph.add((event_node, schema.name, Literal(event, datatype=XSD.sring)))
            graph.add((event_node, schema.startDate, Literal(year, datatype=XSD.date)))
            if venue not in venues:
                venue_node = URIRef(parse.quote(venue))
                graph.add((venue_node, RDF.type, schema.Place))
                graph.add(
                    (venue_node, schema.name, Literal(venue, datatype=XSD.string))
                )
                venues[venue] = venue_node
            graph.add((event_node, schema.location, venues[venue]))
            events[event] = event_node
        graph.add((article, schema.locationCreated, events[event]))
    return graph


def convert_products_to_rdf(
    csv_file: str, name_attr: str = "name", encoding: str = "iso8859"
) -> Graph:
    """
    Converts each product described in the csv as a subgraph in rdf:
                                 description
                               /
                        Product -- name
                        /     \
           name -- Brand     Offer -- PriceSpecification

    Expects a csv file with the following columns:
        id,
        name/title,
        description,
        manufacturer,
        price
    :param csv_file: The file name of the csv file to be used
    :param name_attr: The name of the "name" attribute
    :param encoding: The character encoding of the file
    :return: The entire Knowledge Graph for this file
    """
    df = pd.read_csv(csv_file, encoding=encoding, keep_default_na=False)
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
        brand_node = URIRef(parse.quote(brand_name))
        graph.add((brand_node, RDF.type, schema.Brand))
        graph.add((brand_node, schema.name, Literal(brand_name, datatype=XSD.string)))
        brands[brand_name] = brand_node
    return brands[brand_name]


def _get_price_node(graph: Graph, price: Any) -> Node:
    price_node = BNode()
    price = str(price).replace("$", "").strip()
    graph.add((price_node, RDF.type, schema.PriceSpecification))
    graph.add((price_node, schema.price, Literal(price, datatype=XSD.float)))
    graph.add((price_node, schema.priceCurrency, Literal("USD", datatype=XSD.string)))
    return price_node
