from abc import ABC, abstractmethod
from typing import List

from openea.modules.load.kgs import KGs

money_types = {
    "<http://dbpedia.org/datatype/australianDollar>",
    "<http://dbpedia.org/datatype/bangladeshiTaka>",
    "<http://dbpedia.org/datatype/canadianDollar>",
    "<http://dbpedia.org/datatype/colombianPeso>",
    "<http://dbpedia.org/datatype/croatianKuna>",
    "<http://dbpedia.org/datatype/euro>",
    "<http://dbpedia.org/datatype/hongKongDollar>",
    "<http://dbpedia.org/datatype/indianRupee>",
    "<http://dbpedia.org/datatype/japaneseYen>",
    "<http://dbpedia.org/datatype/norwegianKrone>",
    "<http://dbpedia.org/datatype/poundSterling>",
    "<http://dbpedia.org/datatype/russianRouble>",
    "<http://dbpedia.org/datatype/southKoreanWon>",
    "<http://dbpedia.org/datatype/swedishKrona>",
    "<http://dbpedia.org/datatype/swissFranc>",
    "<http://dbpedia.org/datatype/usDollar>",
}

date_types = {
    "<http://www.w3.org/2001/XMLSchema#date>",
    "<http://www.w3.org/2001/XMLSchema#gYear>",
    "<http://www.w3.org/2001/XMLSchema#gYearMonth>",
}

number_types = {
    "<http://www.w3.org/2001/XMLSchema#double>",
    "<http://www.w3.org/2001/XMLSchema#decimal>",
    "<http://www.w3.org/2001/XMLSchema#float>",
    "<http://www.w3.org/2001/XMLSchema#integer>",
    "<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>",
    "<http://www.w3.org/2001/XMLSchema#positiveInteger>",
}


class AlignedAttribute:
    def __init__(self, k1, v1, k2, v2, measures):
        self.k1 = k1
        self.v1 = v1
        self.k2 = k2
        self.v2 = v2
        self.measures = measures


class AttributeFeatureCombination(ABC):
    def __init__(self, all_measures):
        self.all_measures = all_measures

    @abstractmethod
    def align_attributes(self, e1, e2) -> List[AlignedAttribute]:
        pass


class AllToOneCombination(AttributeFeatureCombination):
    def __init__(self, kgs: KGs, string_measures):
        super().__init__(string_measures)
        self.kgs = kgs
        self.measures = string_measures

    def align_attributes(self, e1, e2) -> List[AlignedAttribute]:
        e1_attrs = sorted(self.kgs.kg1.av_dict[e1].items())
        e2_attrs = sorted(self.kgs.kg2.av_dict[e2].items())

        v1 = " ".join(_remove_type(v) for _, v in e1_attrs)
        v2 = " ".join(_remove_type(v) for _, v in e2_attrs)
        return [AlignedAttribute("all", v1, "all", v2, self.measures)]


class CartesianCombination(AttributeFeatureCombination):
    def __init__(self, kgs: KGs, number_measures, date_measures, string_measures):
        super().__init__(number_measures + date_measures + string_measures)
        self.kgs = kgs
        self.number_measures = number_measures
        self.date_measures = date_measures
        self.string_measures = string_measures

    def align_attributes(self, e1, e2) -> List[AlignedAttribute]:
        """
        Aligns the attributes of the given entities.
        :param e1: id of entity 1
        :param e2: id of entity 2
        :return: tuples of attribute indices
        """
        # add common keys
        kg1_dict = self.kgs.kg1.av_dict
        kg2_dict = self.kgs.kg2.av_dict
        e1_attrs = kg1_dict[e1] if e1 in kg1_dict else kg2_dict.get(e1, None)
        e2_attrs = kg2_dict[e2] if e2 in kg2_dict else kg1_dict.get(e2, None)

        if e1_attrs is None or e2_attrs is None:
            return []
        aligned = {
            (k1, v1, k2, v2)
            for k1, v1 in e1_attrs
            for k2, v2 in e2_attrs
            if k1 == k2 or (v1 != "" and v1 == v2) or _has_same_type(v1, v2)
        }
        return [
            AlignedAttribute(
                k1, _remove_type(v1), k2, _remove_type(v2), self._get_type_measures(v1)
            )
            for (k1, v1, k2, v2) in aligned
        ]

    def _get_type_measures(self, attr):
        datatype = "string" if "^^" not in attr else attr.split("^^")[1]
        if datatype in money_types or datatype in number_types:
            # TODO money conversion probably would be overkill?
            return self.number_measures
        elif datatype in date_types:
            return self.date_measures
        else:
            return self.string_measures


def _has_same_type(v1, v2):
    type1 = "string" if "^^" not in v1 else v1.split("^^")[1]
    type2 = "string" if "^^" not in v2 else v2.split("^^")[1]
    return (
        type1 == type2
        or (type1 in number_types and type2 in number_types)
        or (type1 in date_types and type2 in date_types)
        or (type1 in money_types and type2 in money_types)
    )


def _remove_type(attr: str) -> str:
    if "^^" in attr:
        attr = attr.split("^^")[0]
    if attr.startswith('"') and attr.endswith('"'):
        return attr[1:-1]
    return attr
