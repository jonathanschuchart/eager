from similarities import (
    calculate_from_embeddings,
    calculate_attribute_sims,
    align_attributes,
)


def test_align_attrs():
    e1_attrs = {
        1: "123",
        3: '"123"^^<http://www.w3.org/2001/XMLSchema#double>',
        5: "test",
    }
    e2_attrs = {
        1: "123",
        4: '"123"^^<http://www.w3.org/2001/XMLSchema#double>',
        5: "other",
    }

    assert [(1, 1), (5, 5)] == align_attributes(e1_attrs, e2_attrs)
