from dataclasses import dataclass
from typing import Optional

__all__ = [
    "Caption",
    "Segment",
    "ThesaurusConcept",
    "EnrichedSegment",
]

@dataclass
class Caption:
    start: float
    end: float
    text: str

@dataclass
class Segment:
    start: float
    end: float
    text: str
    captions: list[Caption]

@dataclass
class ThesaurusConcept:
    uri: str
    name: str
    category: str
    alternate_names: Optional[list[str]]
    description: Optional[str]
    top_concept: list[str]
    narrower: list[str]

@dataclass
class MatchedConcept:
    concept: ThesaurusConcept
    source: Optional[str]
    score: Optional[float]

@dataclass
class EnrichedSegment:
    segment: Segment
    matched_concepts: list[MatchedConcept]