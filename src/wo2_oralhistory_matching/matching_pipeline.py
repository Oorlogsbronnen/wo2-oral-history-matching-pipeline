import json
import time
import os
import warnings
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from .models import EnrichedSegment
from .captions import load_vtt
from .segments import create_segments_from_captions, select_segments_to_be_enriched
from .thesaurus import load_thesaurus
from .embeddings import embed_thesaurus_concepts, embed_segment
from .matching import match_segment_to_thesaurus_based_on_embeddings, match_segment_to_thesaurus_based_on_exact_occurrence, match_segment_topdown, llm_validate_segment_matches
from .metadata import extract_name_from_transcript, add_metadata_to_enriched_segment
from .serialize import serialize_enriched_segments, serialize_segments

__all__ = [
    "main",
]

def main():
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    
    # Load environment variables from .env and get the .vtt files from the path listed in the variables.
    load_dotenv()
    start = time.time()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")
    data_folder = os.getenv("DATA_FOLDER")
    force_reload = os.getenv("FORCE_RELOAD", "true").lower() == "true"
    max_tokens = int(os.getenv("TOKEN_LIMIT"))
    minutes_per_batch = int(os.getenv("MINUTES_PER_BATCH"))

    folder_path = Path(data_folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path.resolve()}")
    output_path = Path("output/enriched_segments")

    vtt_files = [
        f for f in folder_path.iterdir() if f.is_file() and f.suffix == ".vtt" and not (output_path / f"{f.stem}_enriched_segments.json").exists()
    ]

    if not vtt_files:
        print(f"No new .vtt files found in {folder_path}.")

    # Make sure the output folders exist.
    Path("output/segments").mkdir(parents=True, exist_ok=True)
    Path("output/selected_segments").mkdir(parents=True, exist_ok=True)
    Path("output/enriched_segments").mkdir(parents=True, exist_ok=True)
    
    # Load and process the WO2-thesaurus.
    thesaurus_concepts = load_thesaurus(force_reload=force_reload)
    descriptive_concepts = [c for c in thesaurus_concepts if c.category == "other" and c.description]
    descriptive_top_concepts = [c for c in thesaurus_concepts if c.category == "other" and c.top_concept]
    camp_and_location_concepts = [c for c in thesaurus_concepts if c.category in ("camp", "location")]
    embedded_thesaurus_concepts = embed_thesaurus_concepts(descriptive_concepts, force_reload=force_reload)

    for i, vtt_file in enumerate(tqdm(vtt_files, desc="Processing VTT-files")):
        # Load the captions from a vtt file.
        captions = load_vtt(vtt_file)
        
        # Collecting the name from the first captions.
        name = extract_name_from_transcript(captions, model = model, api_key = api_key)

        # Create segments based on the captions.
        segments = create_segments_from_captions(captions, model=model, api_key=api_key, minutes_per_batch = minutes_per_batch)

        # Output the segments to a json file.
        output_data = serialize_segments(segments)
        output_path = Path("output/segments") / f"{vtt_file.stem}_segments.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # Determine which segments contain usefull content for a shorter video format.
        selected_segments = select_segments_to_be_enriched(segments, model=model, api_key=api_key, max_tokens=max_tokens)
        output_data = serialize_segments(selected_segments)
        selected_output_path = Path("output/selected_segments") / f"{vtt_file.stem}_selected_segments.json"
        with open(selected_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # Add WO2 thesaurus tags to the selected segments.
        enriched_segments = []
        for segment in selected_segments:
            embedded_segment = embed_segment(segment)
            embedding_matches = match_segment_to_thesaurus_based_on_embeddings(embedded_segment, embedded_thesaurus_concepts, descriptive_concepts, top_k=10)
            exact_matches = match_segment_to_thesaurus_based_on_exact_occurrence(segment, camp_and_location_concepts)
            non_validated_matches = embedding_matches + exact_matches
            validated_matches = llm_validate_segment_matches(segment, non_validated_matches, api_key=api_key, model=model)
            top_down_matches = match_segment_topdown(segment=segment, concepts=descriptive_concepts, top_concepts = descriptive_top_concepts, api_key=api_key, model=model, max_tokens=max_tokens)
            all_matches = top_down_matches + validated_matches
            enriched_segments.append(EnrichedSegment(segment=segment, matched_concepts=all_matches))
    
        # Output the enriched segments to a json file.
        segments_data = serialize_enriched_segments(enriched_segments, name)
        segments_with_metadata = add_metadata_to_enriched_segment(segments_data, api_key=api_key, model=model)
        enriched_output_path = Path("output/enriched_segments") / f"{vtt_file.stem}_enriched_segments.json"
        with open(enriched_output_path, "w", encoding="utf-8") as f:
            json.dump(segments_with_metadata, f, ensure_ascii=False, indent=2)
    
    end = time.time()
    elapsed = end - start
    if elapsed < 60:
        print(f"Time spend: {elapsed:.2f} seconds")
    else:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"Done - Total time spend: {minutes} min {seconds:.0f} sec")