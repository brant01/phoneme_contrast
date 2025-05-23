# src/datasets/parser.py
"""
Enhanced parser that extracts phoneme labels and metadata from file paths.
"""

from pathlib import Path
import torchaudio
from typing import List, Dict, Tuple, Optional
import re
import logging

def extract_phoneme_label(file_path: Path) -> str:
    """
    Extract phoneme label from filename.
    
    Examples:
    - "da.wav" -> "da"
    - "da (short).wav" -> "da"
    - "da1.wav" -> "da"
    - "da1 (short).wav" -> "da"
    - "ada2.wav" -> "ada"
    """
    name = file_path.stem.lower()
    
    # Remove variations like "(short)", "(short version)"
    name = re.sub(r'\s*\([^)]*\)', '', name)
    
    # Remove trailing numbers
    name = re.sub(r'\d+$', '', name)
    
    # Extract alphabetic phoneme part
    match = re.match(r'^([a-z]+)', name)
    if not match:
        raise ValueError(f"Cannot extract label from: {file_path.name}")
    
    return match.group(1)

def extract_metadata(file_path: Path) -> Dict[str, str]:
    """
    Extract metadata from file path structure.
    
    Expected structures:
    - CV: .../CV/Male/_a_/da.wav
    - VCV: .../VCV/Female/ada2.wav
    """
    parts = file_path.parts
    metadata = {
        "structure": "unknown",  # CV or VCV
        "gender": "unknown",     # male or female
        "vowel_context": "unknown",  # _a_, _e_, _i_, _u_ (for CV only)
        "full_path": str(file_path),
        "filename": file_path.name
    }
    
    # Find indices of key parts
    for i, part in enumerate(parts):
        if part.upper() in ["CV", "VCV"]:
            metadata["structure"] = part.upper()
            
            # Gender should be next
            if i + 1 < len(parts):
                gender = parts[i + 1].lower()
                if gender in ["male", "female"]:
                    metadata["gender"] = gender
                    
            # For CV, check for vowel context
            if part.upper() == "CV" and i + 2 < len(parts):
                vowel_context = parts[i + 2]
                if re.match(r'^_[aeiou]_$', vowel_context):
                    metadata["vowel_context"] = vowel_context
            
    # Detect if it's a short version
    metadata["is_short"] = "short" in file_path.stem.lower()
    
    return metadata

def parse_dataset(
    data_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Path], List[int], Dict[str, int], List[Dict]]:
    """
    Parse dataset with metadata.
    
    Returns:
        - file_paths: List of Path objects
        - int_labels: List of integer labels
        - label_map: Dict mapping phoneme strings to integers
        - metadata: List of metadata dicts for each file
    """
    def log(msg: str, level="info"):
        if logger:
            getattr(logger, level)(msg)
        else:
            print(msg)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    file_paths = []
    string_labels = []
    metadata_list = []

    # Find all wav files
    wav_files = list(data_dir.rglob("*.wav"))
    log(f"Found {len(wav_files)} .wav files")

    for wav_file in wav_files:
        try:
            # Extract phoneme label
            label = extract_phoneme_label(wav_file)
            
            # Extract metadata
            metadata = extract_metadata(wav_file)
            metadata["phoneme"] = label  # Add phoneme to metadata
            
            file_paths.append(wav_file)
            string_labels.append(label)
            metadata_list.append(metadata)
            
        except Exception as e:
            log(f"Skipping file: {wav_file.name} — {e}", level="warning")
            continue

    # Create label mapping
    unique_labels = sorted(set(string_labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = [label_map[label] for label in string_labels]
    
    # Log summary
    log(f"Successfully parsed {len(file_paths)} files")
    log(f"Found {len(unique_labels)} unique phonemes: {unique_labels}")
    
    # Count by structure and gender
    cv_count = sum(1 for m in metadata_list if m["structure"] == "CV")
    vcv_count = sum(1 for m in metadata_list if m["structure"] == "VCV")
    male_count = sum(1 for m in metadata_list if m["gender"] == "male")
    female_count = sum(1 for m in metadata_list if m["gender"] == "female")
    
    log(f"Structure: CV={cv_count}, VCV={vcv_count}")
    log(f"Gender: Male={male_count}, Female={female_count}")
    
    # Count vowel contexts for CV
    vowel_counts = {}
    for m in metadata_list:
        if m["structure"] == "CV" and m["vowel_context"] != "unknown":
            vowel = m["vowel_context"]
            vowel_counts[vowel] = vowel_counts.get(vowel, 0) + 1
    if vowel_counts:
        log(f"CV vowel contexts: {vowel_counts}")

    return file_paths, int_labels, label_map, metadata_list