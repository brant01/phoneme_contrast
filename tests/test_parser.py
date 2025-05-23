# tests/test_parser.py
import pytest
from pathlib import Path
from src.datasets.parser import parse_dataset, extract_phoneme_label, extract_metadata


class TestPhonemeExtraction:
    """Test phoneme label extraction from filenames."""
    
    @pytest.mark.parametrize("filename,expected", [
        ("da.wav", "da"),
        ("da (short).wav", "da"),
        ("da1.wav", "da"),
        ("da1 (short).wav", "da"),
        ("ada2.wav", "ada"),
        ("bi (short version).wav", "bi"),
        ("apa.wav", "apa"),
    ])
    def test_extract_phoneme_label(self, filename, expected):
        """Test extraction of phoneme labels from various filename formats."""
        path = Path(filename)
        result = extract_phoneme_label(path)
        assert result == expected
    
    def test_extract_phoneme_label_invalid(self):
        """Test that invalid filenames raise ValueError."""
        with pytest.raises(ValueError):
            extract_phoneme_label(Path("123.wav"))
        
        with pytest.raises(ValueError):
            extract_phoneme_label(Path("(short).wav"))


class TestMetadataExtraction:
    """Test metadata extraction from file paths."""
    
    @pytest.mark.parametrize("path_str,expected", [
        (
            "data/raw/New Stimuli 9-8-2024/CV/Male/_a_/da.wav",
            {"structure": "CV", "gender": "male", "vowel_context": "_a_", "is_short": False}
        ),
        (
            "data/raw/New Stimuli 9-8-2024/CV/Female/_i_/bi (short version).wav",
            {"structure": "CV", "gender": "female", "vowel_context": "_i_", "is_short": True}
        ),
        (
            "data/raw/New Stimuli 9-8-2024/VCV/Female/ada2.wav",
            {"structure": "VCV", "gender": "female", "vowel_context": "unknown", "is_short": False}
        ),
    ])
    def test_extract_metadata(self, path_str, expected):
        """Test metadata extraction from different path structures."""
        path = Path(path_str)
        metadata = extract_metadata(path)
        
        for key, value in expected.items():
            assert metadata[key] == value


class TestFullParser:
    """Test the complete parsing function."""
    
    @pytest.fixture
    def data_dir(self):
        """Get the data directory path."""
        return Path("data/raw/New Stimuli 9-8-2024")
    
    def test_parse_dataset(self, data_dir):
        """Test parsing the full dataset."""
        if not data_dir.exists():
            pytest.skip(f"Data directory not found: {data_dir}")
        
        file_paths, int_labels, label_map, metadata_list = parse_dataset(data_dir)
        
        # Basic assertions
        assert len(file_paths) > 0
        assert len(file_paths) == len(int_labels)
        assert len(file_paths) == len(metadata_list)
        assert len(label_map) > 0
        
        # All labels should be in range
        assert all(0 <= label < len(label_map) for label in int_labels)
        
        # All phonemes in metadata should be in label_map
        for metadata in metadata_list:
            assert metadata["phoneme"] in label_map
        
        # Check metadata structure
        for metadata in metadata_list:
            assert metadata["structure"] in ["CV", "VCV"]
            assert metadata["gender"] in ["male", "female"]
            if metadata["structure"] == "CV":
                assert metadata["vowel_context"] in ["_a_", "_e_", "_i_", "_u_"]
            else:
                assert metadata["vowel_context"] == "unknown"
    
    def test_parse_dataset_missing_dir(self):
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_dataset(Path("nonexistent/directory"))