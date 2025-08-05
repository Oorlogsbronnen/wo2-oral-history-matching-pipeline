# WO2 Oral History Matching Pipeline

This project contains a Python pipeline for automatically segmenting World War II–related oral history interviews (in `.vtt` format) and enriching the segments with concepts from a WWII thesaurus.

## Features

- **Transcript processing**: loads `.vtt` files containing interview transcripts  
- **Segmentation**: splits transcripts into meaningful segments  
- **Selection**: determines which segments are most relevant for short‑form presentation  
- **Thesaurus matching**: links segments to WWII concepts via embeddings, exact matches, and LLM validation
- **Storage**: outputs results as JSON files, including metadata

## Installation

1. **Clone the repository and navigate to the root of the repository**

2. **Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install relevant packages**
```bash
pip install .
```

## Configuration

Copy the example environment file and rename it to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and fill in the correct API keys and settings.

Key variables:
- `OPENAI_API_KEY`: API key for OpenAI
- `MODEL`: LLM model to use
- `TOKEN_LIMIT`: maximum number of tokens for LLM tasks
- `FORCE_RELOAD`: whether to reload the WWII thesaurus
- `DATA_FOLDER`: folder containing `.vtt` files
- `MINUTES_PER_BATCH`: batch size in minutes

## Directory Structure

```
interview_transcripts/             # Input folder for .vtt files
output/                            # Output folder for results
scripts/                           # Runner scripts
src/wo2_oralhistory_matching/      # Python package with pipeline and modules
.env.example                       # Example configuration file
.gitignore
pyproject.toml
```

The `interview_transcripts` and `output` folders contain a `.gitkeep` file so they remain in the repository, while actual data is ignored via `.gitignore`.

## Usage

Place `.vtt` files in the folder defined by `DATA_FOLDER` in `.env` (default: `interview_transcripts`).

Run the pipeline via the runner script:

```bash
python scripts/run_pipeline.py
```

The results will be stored in `output/`:
- `output/segments`: JSON with all segments
- `output/selected_segments`: JSON with selected segments
- `output/enriched_segments`: JSON with enriched segments

## Requirements

All requirements are defined in `pyproject.toml` and will be installed automatically:
- numpy
- torch
- openai
- transformers (<4.55.0)
- rdflib
- platformdirs
- scikit-learn
- tiktoken
- ftfy
- python-dotenv
- tqdm
