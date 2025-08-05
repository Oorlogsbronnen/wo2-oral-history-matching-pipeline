import os
from transformers import AutoTokenizer, AutoModel
from platformdirs import user_cache_dir

MODEL_NAME = "intfloat/multilingual-e5-base"
APP_NAME = "wo2-oralhistory-matching"
MODEL_CACHE_DIR = os.path.join(user_cache_dir(APP_NAME), "model")

def _download_model():
    """
    Download the model used for embeddings.
    """
    AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    AutoModel.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)