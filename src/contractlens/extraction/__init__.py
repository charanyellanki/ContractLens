"""Extraction module for clause extraction from contracts."""

from contractlens.extraction.extractor import ClauseExtractor
from contractlens.extraction.prompts import EXTRACTION_PROMPTS

__all__ = ["ClauseExtractor", "EXTRACTION_PROMPTS"]