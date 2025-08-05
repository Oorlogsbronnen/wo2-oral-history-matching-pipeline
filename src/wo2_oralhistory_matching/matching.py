import numpy as np
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
from .prompts import _build_match_validation_prompt, _build_topdown_matching_prompt
from .response_cleaner import _clean_json_output
from .chat_router import _chat
from .batching import _batch_concept_labels_by_tokens
from .models import ThesaurusConcept, MatchedConcept, Segment

__all__ = [
    "match_segment_to_thesaurus_based_on_embeddings",
    "match_segment_topdown",
    "match_segment_to_thesaurus_based_on_exact_occurrence"
    "deduplicate_matches",
    "llm_validate_segment_matches"
]

def _generate_concept_labels(concepts: list[ThesaurusConcept]) -> list[str]:
    """
    Create a label for a thesaurus concept containing both name and description.
    """
    concept_labels = [
        f"{c.name} – {c.description}" if c.description else c.name
        for c in concepts
    ]

    return concept_labels

def _generate_matched_concept_labels(concepts: list[MatchedConcept]) -> list[str]:
    """
    Create a label for a matched thesaurus concept containing both name and description.
    """
    concept_labels = [
        f"{c.concept.name} – {c.concept.description}" if c.concept.description else c.concept.name
        for c in concepts
    ]

    return concept_labels

def _find_narrower_concepts(current_concepts: list[ThesaurusConcept], target_concepts: list[ThesaurusConcept]) -> list[ThesaurusConcept]:
    """
    Helper function used to find narrower terms for a list of thesaurus concepts.
    """
    narrower_concept_uris = []
    for concept in current_concepts:
        narrower_concept_uris.extend(concept.narrower)
    
    narrower_concepts = [c for c in target_concepts if c.uri in narrower_concept_uris]
    return narrower_concepts

def _extract_selected_names(parsed):
    """
    Helper: Extract concept names and optional scores from the parsed LLM output.
    Returns a dict: { "concept_name": score or None }
    """
    selected = {}
    for item in parsed:
        if isinstance(item, dict) and "concept" in item:
            name = item["concept"].strip()
            score = item.get("score")
            try:
                score_val = float(score) if score is not None else None
            except (TypeError, ValueError):
                score_val = None
            selected[name] = score_val
        elif isinstance(item, str):
            selected[item.strip()] = None
    return selected

def _get_matched_concepts_from_response(response, concepts: list[ThesaurusConcept]) -> list[MatchedConcept]:
    """
    Helper function used to parse the LLM response into a list of relevant matched concepts.
    """
    clean_response = _clean_json_output(response)
    try:
        parsed = json.loads(clean_response)
    except json.JSONDecodeError:
        print("An error occurred while parsing the JSON-LLM output. No matches were validated.")
        print("RAW response:", repr(clean_response))
        return []

    if not isinstance(parsed, list):
        print("Unexpected output (no list):", parsed)
        return []
    
    name_with_score = _extract_selected_names(parsed)
    if not name_with_score:
        return []
    
    matched = []
    for c in concepts:
        if c.name.strip() in name_with_score:
            matched.append(
                MatchedConcept(
                    concept=c,
                    source='Top-down matching',
                    score=name_with_score[c.name.strip()]
                )
            )
    return matched

def _get_revelant_matched_concepts_from_response(response, concepts: list[MatchedConcept]) -> list[MatchedConcept]:
    """
    Parse de LLM response (JSON) en koppel de optionele scores aan de bestaande MatchedConcepts.
    """
    clean_response = _clean_json_output(response)

    try:
        parsed = json.loads(clean_response)
    except json.JSONDecodeError:
        print("An error occurred while parsing the JSON-LLM output. No matches were validated.")
        print("RAW response:", repr(clean_response))
        return []

    if not isinstance(parsed, list):
        print("Unexpected output (no list):", parsed)
        return []

    name_to_score = {}
    for item in parsed:
        if isinstance(item, dict) and "concept" in item:
            name = item["concept"].strip()
            score_val = None
            score = item.get("score")
            if score is not None:
                try:
                    score_val = float(score)
                except (TypeError, ValueError):
                    score_val = None
            name_to_score[name] = score_val
        elif isinstance(item, str):
            name_to_score[item.strip()] = None

    if not name_to_score:
        return []

    validated_matches = []
    for mc in concepts:
        concept_name = mc.concept.name.strip()
        if concept_name in name_to_score:
            new_score = name_to_score[concept_name]
            validated_matches.append(
                MatchedConcept(
                    concept=mc.concept,
                    source=mc.source,
                    score=new_score if new_score is not None else mc.score
                )
            )

    return validated_matches

def _safe_get_relevant_concepts(prompt, api_key, model, top_concepts, max_retries=1):
    """
    Adds the option to retry finding relevant concepts.
    """
    import json

    for attempt in range(max_retries + 1):
        response = _chat(prompt, api_key=api_key, model=model)
        try:
            return _get_matched_concepts_from_response(response, top_concepts)
        except (TypeError, json.JSONDecodeError) as e:
            print(f"Parsing error (attempt {attempt+1}): {e}")
            if attempt == max_retries:
                raise
            print("Retrying request to LLM...")
    return []

def deduplicate_matches(matches):
    seen_uris = set()
    deduped = []
    for m in matches:
        if m.concept.uri not in seen_uris:
            deduped.append(m)
            seen_uris.add(m.concept.uri)
    return deduped

def match_segment_topdown(segment: Segment, concepts: list[ThesaurusConcept], top_concepts: list[ThesaurusConcept], api_key:str, model: str = "gpt-4.1", max_tokens: int = 800000) -> list[MatchedConcept]:
    """
    Finds all relevant thesaurusconcepts based on top-down LLM-matching in the thesaurus.
    """

    matches = []

    #Step 1: Evaluate top-concepts for relevant schemes.
    top_concept_labels = _generate_concept_labels(top_concepts)
    batched_labels = _batch_concept_labels_by_tokens(top_concept_labels, segment.text, model=model, max_tokens=max_tokens)
    relevant_top_matches = []

    for batch_labels in batched_labels:
        top_concept_prompt = _build_topdown_matching_prompt(concept_labels=batch_labels, segment_text=segment.text)
        relevant_batch_concepts = _safe_get_relevant_concepts(top_concept_prompt, api_key, model, top_concepts)
        relevant_top_matches.extend(relevant_batch_concepts)

    if not relevant_top_matches:
        return []

    #Step 2: Recursively go through all narrower concepts.
    current_matches = relevant_top_matches
    seen_uris = {m.concept.uri for m in relevant_top_matches}

    while current_matches:
        current_concepts = [m.concept for m in current_matches]
        narrower_concepts = _find_narrower_concepts(current_concepts=current_concepts, target_concepts=concepts)
        new_narrower_concepts = [c for c in narrower_concepts if c.uri not in seen_uris]
        
        if not new_narrower_concepts:
            break
        
        narrower_labels = _generate_concept_labels(new_narrower_concepts) 
        batched_narrower_labels = _batch_concept_labels_by_tokens(narrower_labels, segment.text, model=model, max_tokens=max_tokens)
        relevant_narrower_matches = []
        
        for batch_labels in batched_narrower_labels:
            narrower_concept_prompt = _build_topdown_matching_prompt(concept_labels=batch_labels, segment_text=segment.text)
            relevant_batch_concepts = _safe_get_relevant_concepts(narrower_concept_prompt, api_key, model, new_narrower_concepts)
            relevant_narrower_matches.extend(relevant_batch_concepts)
        
        if not relevant_narrower_matches:
            break
        
        seen_uris.update(m.concept.uri for m in relevant_narrower_matches)
        matches.extend(relevant_narrower_matches)
        current_matches = relevant_narrower_matches
    return matches

def match_segment_to_thesaurus_based_on_exact_occurrence(segment: Segment, concepts: list[ThesaurusConcept]) -> list[MatchedConcept]:
    """
    Finds relevant thesaurus concepts based on exact occurrence of the full concept name in the text.
    Only counts exact matches (case-insensitive, word boundaries).
    """
    matches = []
    text = segment.text.lower()

    for concept in concepts:
        pattern = r'\b' + re.escape(concept.name.lower()) + r'\b'
        if re.search(pattern, text):
            matches.append(
                MatchedConcept(
                    concept=concept,
                    source='Exact occurrence',
                    score=1.0
                )
            )
    return matches

def match_segment_to_thesaurus_based_on_embeddings(embedded_segment: np.ndarray, embedded_concepts: np.ndarray, concepts: list[ThesaurusConcept], top_k: int = 20) -> list[MatchedConcept]:
    """
    Search the top_k most relevant thesaurusconcepts for a segment based on the embeddings.
    """
    similarities = cosine_similarity(
        embedded_segment.reshape(1, -1), embedded_concepts
    )[0]
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [MatchedConcept(concept=concepts[i], source='Embedding similarities', score=float(similarities[i])) for i in top_indices]

def llm_validate_segment_matches(segment: Segment, matched_concepts: list[MatchedConcept], api_key: str, model: str = "gpt-4.1") -> list[MatchedConcept]:
    """
    Let an LLM validate the matches generated.
    """

    concept_labels = _generate_matched_concept_labels(matched_concepts)

    prompt = _build_match_validation_prompt(segment.text, concept_labels)
    response = _chat(prompt, api_key=api_key, model=model)
    validated_concepts =  _get_revelant_matched_concepts_from_response(response, matched_concepts)

    return validated_concepts