# scripts/explore_data.py
from pathlib import Path
import os

def explore_directory(data_path):
    """Explore the directory structure and list all files."""
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return
    
    print(f"Exploring: {data_dir}\n")
    
    # Get all .wav files
    wav_files = list(data_dir.rglob("*.wav"))
    print(f"Total .wav files found: {len(wav_files)}\n")
    
    # Show first 10 files with their full path structure
    print("First 10 files:")
    for i, wav_file in enumerate(wav_files[:10]):
        # Get relative path from data_dir
        rel_path = wav_file.relative_to(data_dir)
        print(f"{i+1}. {rel_path}")
    
    print("\nDirectory structure:")
    # Show unique directory patterns
    dir_patterns = set()
    for wav_file in wav_files:
        rel_path = wav_file.relative_to(data_dir)
        # Get directory part only
        dir_part = str(rel_path.parent)
        dir_patterns.add(dir_part)
    
    for pattern in sorted(dir_patterns):
        count = sum(1 for f in wav_files if str(f.relative_to(data_dir).parent) == pattern)
        print(f"  {pattern}: {count} files")
    
    print("\nUnique filename patterns (first 20):")
    filenames = [f.name for f in wav_files]
    for i, name in enumerate(sorted(set(filenames))[:20]):
        print(f"  {name}")

if __name__ == "__main__":
    explore_directory("data/raw/New Stimuli 9-8-2024")