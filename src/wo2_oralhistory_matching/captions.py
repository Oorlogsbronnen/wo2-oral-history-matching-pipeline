from ftfy import fix_text
from .models import Caption

__all__ = [
    "load_vtt",
]

def load_vtt(vtt_file: str) -> list[Caption]:
    """
    Load a WebVTT subtitle file and return a list of Caption objects.
    """
    captions = []
    
    with open(vtt_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line != 'WEBVTT':
            raise ValueError('File does not start with WEBVTT header')
            
        current_time = None
        current_text = []
        
        def store_current_caption():
            nonlocal current_time, current_text
            if current_time is not None and current_text:
                start_time, end_time = _parse_timestamp(current_time)
                text = fix_text('\n'.join(current_text))
                caption = Caption(start=start_time, end=end_time, text=text)
                captions.append(caption)
                current_time = None
                current_text = []
        
        for line in f:
            line = line.strip()
            
            if not line:
                store_current_caption()
                continue
                
            if '-->' in line:
                current_time = line
            else:
                current_text.append(line)
                
        store_current_caption()
            
    return captions

def _parse_timestamp(timestamp_line: str) -> tuple[float, float]:
    """
    Parse a timestamp line into start and end times in seconds.
    """
    start, end = timestamp_line.split('-->')
    return (
        _convert_timestamp_to_seconds(start.strip()),
        _convert_timestamp_to_seconds(end.strip())
    )

def _convert_timestamp_to_seconds(timestamp: str) -> float:
    """
    Parse a timestamp to seconds.
    """
    parts = [int(p) for p in timestamp.replace('.', ':').split(':')]
    parts = [0] * (4 - len(parts)) + parts
    hours, minutes, seconds, milliseconds = parts
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000