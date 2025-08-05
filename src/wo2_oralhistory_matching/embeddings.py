import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from platformdirs import user_cache_dir
from .models import ThesaurusConcept, Segment
from .download_model import MODEL_NAME, MODEL_CACHE_DIR, _download_model

__all__ = [
    "embed_thesaurus_concepts",
    "embed_segment",
]

_tokenizer = None
_model = None

APP_NAME = "wo2-oralhistory-matching"
CACHE_DIR = user_cache_dir(APP_NAME)
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
CONCEPT_EMBEDDINGS_PATH = os.path.join(EMBEDDING_CACHE_DIR, "concept_embeddings.npy")
CONCEPT_OBJECTS_PATH = os.path.join(EMBEDDING_CACHE_DIR, "concept_objects.pkl")

def _get_model():
    """
    Get the model from cache or download the model.
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _download_model()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
        _model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
        _model.to("cpu")
    return _tokenizer, _model

def _embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Create embeddings for a list of strings.
    """
    tokenizer, model = _get_model()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_embeddings = outputs.last_hidden_state * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            lengths = attention_mask.sum(dim=1)
            embeddings = sum_embeddings / lengths
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # cosine-ready
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()

def _concept_to_text(concept: ThesaurusConcept) -> str:
    """
    Create a label for a thesaurus concept containing both name and description.
    """
    parts = [concept.name]
    if concept.alternate_names:
        parts.append(" / ".join(concept.alternate_names))
    if concept.description:
        parts.append(concept.description)
    return " | ".join(parts)

def _embed_concepts(concepts: list[ThesaurusConcept]) -> np.ndarray:
    """
    Create embeddings for a list of thesaurus concepts.
    """
    texts = [_concept_to_text(c) for c in concepts]
    return _embed_texts(texts)

def embed_segment(segment: Segment, batch_size: int = 1) -> np.ndarray:
    """
    Create an embedding for a segment.
    """
    return _embed_texts([segment.text], batch_size=batch_size)[0]

def embed_thesaurus_concepts(concepts: list[ThesaurusConcept], force_reload: bool = False) -> np.ndarray:
    """
    Create embeddings for a list of thesaurus concepts or load the embeddings from cache.
    """
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    if not force_reload and os.path.exists(CONCEPT_EMBEDDINGS_PATH):
        return np.load(CONCEPT_EMBEDDINGS_PATH)
    embeddings = _embed_concepts(concepts).astype("float32")
    np.save(CONCEPT_EMBEDDINGS_PATH, embeddings)
    return embeddings