import os
import pickle
from platformdirs import user_cache_dir
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, SKOS, Namespace
from .models import ThesaurusConcept

__all__ = [
    "load_thesaurus"
]

SCHEMA = Namespace("http://schema.org/")
WGS = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
NIOD = Namespace("https://data.niod.nl/thesaurus_wo2/")

APP_NAME = "wo2-oralhistory-matching"
CACHE_DIR = user_cache_dir(APP_NAME)
CACHE_CONCEPTS_PATH = os.path.join(CACHE_DIR, "wo2_concepts.pkl")

THESAURUS_URL = "https://data.spinque.com/ld/data/oorlogsbronnen/wo2_thesaurus/data/export.nt"
EXCLUDE_PREDICATE_OORLOGDICHTBIJ = URIRef("https://data.niod.nl/thesaurus_wo2/ImagesWW2/oorlogDichtbijConcept")

def _load_thesaurus_from_web() -> list[ThesaurusConcept]:
    """
    Downdloads the WO2 thesaurus and processes the classes.
    Skips concepts where oorlogDichtbijConcept is explicitly false and that are not top_concepts.
    """
    print("\nDownloading WO2 thesaurus from:", THESAURUS_URL, "...")
    g = Graph()
    g.parse(THESAURUS_URL, format="nt")

    concepts = []
    skipped_flagged = 0

    for s in g.subjects(RDF.type, SKOS.Concept):
        in_schemes = [str(o) for o in g.objects(s, SKOS.inScheme)]
        top_concept = [str(o) for o in g.objects(s, SKOS.topConceptOf)]
        narrower = [str(o) for o in g.objects(s, SKOS.narrower)]

        if (any(str(o).strip().lower() == "false" for o in g.objects(s, EXCLUDE_PREDICATE_OORLOGDICHTBIJ))) and not top_concept:
            skipped_flagged += 1
            continue
        if "https://data.niod.nl/WO2_Thesaurus/11183" in in_schemes:
            skipped_flagged += 1
            continue

        uri = str(s)
        name = next(
            (str(o) for o in g.objects(s, SKOS.prefLabel) if getattr(o, "language", None) == "nl"),
            ""
        )

        if "https://data.niod.nl/WO2_Thesaurus/kampen/3650" in in_schemes:
            category = "camp"
        elif "https://data.niod.nl/WO2_Thesaurus/6564" in in_schemes:
            category = "location"
        else:
            category = "other"

        alt_labels = [str(o) for o in g.objects(s, SKOS.altLabel)]
        description = str(g.value(s, SKOS.scopeNote))

        concept = ThesaurusConcept(
            uri=uri,
            name=name,
            category=category,
            alternate_names=alt_labels if alt_labels else None,
            description=description if description else None,
            top_concept= top_concept if top_concept else [],
            narrower= narrower if narrower else []
        )
        concepts.append(concept)

    print(f"Thesaurus loaded: {len(concepts)} concepts included.")
    print(f" - Skipped {skipped_flagged} with oorlogDichtbijConcept=false or from Technische Lijsten")
    return concepts

def _save_to_cache(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _load_from_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def load_thesaurus(force_reload=False) -> list[ThesaurusConcept]:
    """
    Load the WO2-thesaurus.
    """
    if not force_reload and os.path.exists(CACHE_CONCEPTS_PATH):
        return _load_from_cache(CACHE_CONCEPTS_PATH)
    
    concepts = _load_thesaurus_from_web()
    _save_to_cache(concepts, CACHE_CONCEPTS_PATH)
    return concepts