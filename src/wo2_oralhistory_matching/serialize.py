__all__ = [
    "serialize_enriched_segments",
    "serialize_segments"
]

def serialize_enriched_segments(enriched_segments, name):
    """
    Used temporarily for exporting the data.
    """
    result = []
    for enriched in enriched_segments:
        segment_data = {
            "interviewee_name": name,
            "start": enriched.segment.start,
            "end": enriched.segment.end,
            "text": enriched.segment.text,
            "matched_concepts": [
                {
                    "uri": mc.concept.uri,
                    "name": mc.concept.name,
                    "source": mc.source,
                    "score": mc.score
                }
                for mc in enriched.matched_concepts
            ]
        }
        result.append(segment_data)
    return result

def serialize_segments(segments):
    """
    Used temporarily for exporting the data.
    """
    result = []
    for segment in segments:
        segment_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        result.append(segment_data)
    return result