import pytest
import json
from assertpy import assert_that
from src.quality.analyze_types import _find_best_general


@pytest.fixture
def superclasses_dict():
    with open("data/OpenEA/typed_links/superclasses.json", "r") as fp:
        return json.load(fp)


@pytest.fixture
def movie_types():
    return [
        "http://www.w3.org/2002/07/owl#Thing",
        "http://www.wikidata.org/entity/Q386724",
        "http://dbpedia.org/ontology/Film",
        "http://dbpedia.org/ontology/Wikidata:Q11424",
        "http://dbpedia.org/ontology/Work",
        "http://schema.org/CreativeWork",
        "http://schema.org/Movie",
        "http://umbel.org/umbel/rc/Movie_CW",
        "http://dbpedia.org/class/yago/WikicatFilmsDirectedByManiRatnam",
        "http://dbpedia.org/class/yago/WikicatIndianDramaFilms",
        "http://dbpedia.org/class/yago/WikicatIndianFilms",
        "http://dbpedia.org/class/yago/WikicatTamil-languageFilms",
        "http://dbpedia.org/class/yago/WikicatTamil-languageFilmsDubbedInTelugu",
        "http://dbpedia.org/class/yago/WikicatTamilFilmsOf2002",
        "http://dbpedia.org/class/yago/Abstraction100002137",
        "http://dbpedia.org/class/yago/Album106591815",
        "http://dbpedia.org/class/yago/Artifact100021939",
        "http://dbpedia.org/class/yago/Creation103129123",
        "http://dbpedia.org/class/yago/EndProduct103287178",
        "http://dbpedia.org/class/yago/Event100029378",
        "http://dbpedia.org/class/yago/Instrumentality103575240",
        "http://dbpedia.org/class/yago/Medium106254669",
        "http://dbpedia.org/class/yago/Movie106613686",
        "http://dbpedia.org/class/yago/Object100002684",
        "http://dbpedia.org/class/yago/Oeuvre103841417",
        "http://dbpedia.org/class/yago/PhysicalEntity100001930",
        "http://dbpedia.org/class/yago/Product104007894",
        "http://dbpedia.org/class/yago/PsychologicalFeature100023100",
        "http://dbpedia.org/class/yago/Show106619065",
        "http://dbpedia.org/class/yago/SocialEvent107288639",
        "http://dbpedia.org/class/yago/Whole100003553",
        "http://dbpedia.org/class/yago/YagoPermanentlyLocatedEntity",
        "http://dbpedia.org/class/yago/Wikicat2000sDramaFilms",
        "http://dbpedia.org/class/yago/Wikicat2002Films",
        "http://dbpedia.org/class/yago/WikicatWorksAboutAdoption",
    ]


@pytest.fixture
def athlete_types():
    return [
        "http://www.w3.org/2002/07/owl#Thing",
        "http://xmlns.com/foaf/0.1/Person",
        "http://dbpedia.org/ontology/Person",
        "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Agent",
        "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#NaturalPerson",
        "http://www.wikidata.org/entity/Q215627",
        "http://www.wikidata.org/entity/Q24229398",
        "http://www.wikidata.org/entity/Q5",
        "http://www.wikidata.org/entity/Q937857",
        "http://dbpedia.org/ontology/Agent",
        "http://dbpedia.org/ontology/Athlete",
        "http://dbpedia.org/ontology/SoccerPlayer",
        "http://schema.org/Person",
        "http://dbpedia.org/class/yago/WikicatKetteringTownF.C.Players",
        "http://dbpedia.org/class/yago/WikicatRushden&DiamondsF.C.Players",
        "http://dbpedia.org/class/yago/WikicatTheFootballLeaguePlayers",
        "http://dbpedia.org/class/yago/Athlete109820263",
        "http://dbpedia.org/class/yago/BasketballPlayer109842047",
        "http://dbpedia.org/class/yago/CausalAgent100007347",
        "http://dbpedia.org/class/yago/Contestant109613191",
        "http://dbpedia.org/class/yago/FootballPlayer110101634",
        "http://dbpedia.org/class/yago/Forward110105733",
        "http://dbpedia.org/class/yago/LivingThing100004258",
        "http://dbpedia.org/class/yago/Object100002684",
        "http://dbpedia.org/class/yago/Organism100004475",
        "http://dbpedia.org/class/yago/Person100007846",
        "http://dbpedia.org/class/yago/PhysicalEntity100001930",
        "http://dbpedia.org/class/yago/Player110439851",
        "http://dbpedia.org/class/yago/Whole100003553",
        "http://dbpedia.org/class/yago/YagoLegalActor",
        "http://dbpedia.org/class/yago/YagoLegalActorGeo",
        "http://dbpedia.org/class/yago/WikicatEnglishFootballers",
        "http://dbpedia.org/class/yago/WikicatAssociationFootballForwards",
        "http://dbpedia.org/class/yago/WikicatBishop'sStortfordF.C.Players",
        "http://dbpedia.org/class/yago/WikicatLivingPeople",
        "http://dbpedia.org/class/yago/WikicatNorthamptonTownF.C.Players",
    ]


@pytest.fixture
def sports_team_types():
    return [
        "http://www.w3.org/2002/07/owl#Thing",
        "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Agent",
        "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#SocialPerson",
        "http://www.wikidata.org/entity/Q24229398",
        "http://www.wikidata.org/entity/Q43229",
        "http://www.wikidata.org/entity/Q4498974",
        "http://dbpedia.org/ontology/Agent",
        "http://dbpedia.org/ontology/HockeyTeam",
        "http://dbpedia.org/ontology/Organisation",
        "http://dbpedia.org/ontology/SoccerClub",
        "http://dbpedia.org/ontology/SportsTeam",
        "http://schema.org/Organization",
        "http://schema.org/SportsTeam",
        "http://dbpedia.org/class/yago/WikicatIceHockeyTeamsInRussia",
        "http://dbpedia.org/class/yago/WikicatKontinentalHockeyLeagueTeams",
        "http://dbpedia.org/class/yago/WikicatSportsClubsEstablishedIn1946",
        "http://dbpedia.org/class/yago/WikicatSportsClubsEstablishedIn1947",
        "http://dbpedia.org/class/yago/Abstraction100002137",
        "http://dbpedia.org/class/yago/Association108049401",
        "http://dbpedia.org/class/yago/Club108227214",
        "http://dbpedia.org/class/yago/Group100031264",
        "http://dbpedia.org/class/yago/HockeyTeam108080386",
        "http://dbpedia.org/class/yago/Organization108008335",
        "http://dbpedia.org/class/yago/SocialGroup107950920",
        "http://dbpedia.org/class/yago/Team108208560",
        "http://dbpedia.org/class/yago/Unit108189659",
        "http://dbpedia.org/class/yago/YagoLegalActor",
        "http://dbpedia.org/class/yago/YagoLegalActorGeo",
        "http://dbpedia.org/class/yago/YagoPermanentlyLocatedEntity",
    ]


@pytest.fixture
def region_types():
    return [
        "http://dbpedia.org/ontology/AdministrativeRegion",
        "http://dbpedia.org/class/yago/WikicatSubdivisionsOfGuinea",
        "http://dbpedia.org/class/yago/WikicatThird-levelAdministrativeCountrySubdivisions",
        "http://dbpedia.org/class/yago/GeographicalArea108574314",
        "http://dbpedia.org/class/yago/Location100027167",
        "http://dbpedia.org/class/yago/Object100002684",
        "http://dbpedia.org/class/yago/PhysicalEntity100001930",
        "http://dbpedia.org/class/yago/Region108630985",
        "http://dbpedia.org/class/yago/Subdivision108674251",
        "http://dbpedia.org/class/yago/Tract108673395",
        "http://dbpedia.org/class/yago/YagoGeoEntity",
        "http://dbpedia.org/class/yago/YagoLegalActorGeo",
        "http://dbpedia.org/class/yago/YagoPermanentlyLocatedEntity",
        "http://dbpedia.org/class/yago/WikicatCountrySubdivisionsOfAfrica",
    ]


def test_movie(movie_types, superclasses_dict):
    type = _find_best_general(movie_types, superclasses_dict)
    assert_that(type).is_equal_to("http://dbpedia.org/ontology/Film")


def test_athlete(athlete_types, superclasses_dict):
    type = _find_best_general(athlete_types, superclasses_dict)
    assert_that(type).is_equal_to("http://dbpedia.org/ontology/Person")


def test_sports_team(sports_team_types, superclasses_dict):
    type = _find_best_general(sports_team_types, superclasses_dict)
    assert_that(type).is_equal_to("http://dbpedia.org/ontology/Organisation")


def test_region(region_types, superclasses_dict):
    type = _find_best_general(region_types, superclasses_dict)
    print(type)
    assert_that(type).is_equal_to("http://dbpedia.org/ontology/Location")
