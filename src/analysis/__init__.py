"""
Comprehensive analysis framework for phoneme contrast experiments.

This module provides publication-ready analysis tools including:
- Statistical significance testing
- Speaker invariance analysis
- Phonetic feature clustering
- Baseline comparisons
- Publication-quality visualizations
"""

from .baseline_comparisons import BaselineComparison
from .phonetic_analysis import PhoneticFeatureAnalyzer
from .significance_testing import SignificanceTester
from .speaker_analysis import SpeakerInvarianceAnalyzer
from .visualization import PublicationVisualizer

__all__ = [
    "SignificanceTester",
    "SpeakerInvarianceAnalyzer",
    "PhoneticFeatureAnalyzer",
    "BaselineComparison",
    "PublicationVisualizer",
]