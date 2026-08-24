"""
Text Cleaning, PII Normalization, and Narrative Validation Engine.
Prepares messy raw CFPB consumer complaint narratives for high-quality embedding generation.
"""

import re
import unicodedata
from typing import Optional


# =============================================================================
# Regular Expressions for CFPB Redactions and Noise
# =============================================================================

# Matches dates like XX/XX/2023, XX/XX/XXXX, 2023-XX-XX
RE_DATE_REDACTION = re.compile(
    r"\b(XX/XX/\d{4}|XX/XX/XXXX|\d{2}/XX/\d{4}|XX-\d{2}-\d{4}|\d{4}-XX-XX)\b",
    re.IGNORECASE
)

# Matches dollar amounts like {$1200.00}, {$XX.XX}, {$50.00}
RE_AMOUNT_REDACTION = re.compile(
    r"\{\$[\d,X\.]+\}",
    re.IGNORECASE
)

# Matches dollar signs followed by XXXX (e.g., $XXXX, $XX,XXX)
RE_DOLLAR_XXXX = re.compile(
    r"\$[\s]*X+[\d,X\.]*",
    re.IGNORECASE
)

# Matches standalone sequences of XXXX (names, accounts, institutions)
RE_GENERIC_XXXX = re.compile(
    r"\bX{2,}\b"
)

# Matches excessive consecutive horizontal whitespace
RE_MULTIPLE_WHITESPACE = re.compile(r"[ \t]+")
# Matches whitespace around newlines
RE_WHITESPACE_NEWLINES = re.compile(r"[ \t]*\n[ \t]*")
# Matches 3 or more consecutive newlines
RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


# =============================================================================
# Core Cleaning Functions
# =============================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalizes Unicode characters (e.g. curly quotes, non-breaking spaces) to standard UTF-8.
    """
    if not text:
        return ""
    
    # Normalize Unicode compatibility (NFKD)
    normalized = unicodedata.normalize("NFKD", text)
    
    # Replace common typography artifacts
    replacements = {
        "\xa0": " ",      # Non-breaking space
        "\u2019": "'",    # Right single quotation mark
        "\u2018": "'",    # Left single quotation mark
        "\u201c": '"',    # Left double quotation mark
        "\u201d": '"',    # Right double quotation mark
        "\u2014": " - ",  # Em dash
        "\u2013": " - ",  # En dash
        "\u2026": "...",  # Ellipsis
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
        
    return normalized


def normalize_pii_redactions(text: str) -> str:
    """
    Replaces raw CFPB redaction markers with clean semantic placeholder tokens.
    
    Examples:
        'On XX/XX/2023' -> 'On [DATE]'
        'charged {$450.00}' -> 'charged [AMOUNT]'
        'called XXXX bank' -> 'called [REDACTED] bank'
    """
    if not text:
        return ""
    
    # 1. Normalize dates first (to avoid generic XXXX clobbering dates)
    text = RE_DATE_REDACTION.sub("[DATE]", text)
    
    # 2. Normalize currency amounts
    text = RE_AMOUNT_REDACTION.sub("[AMOUNT]", text)
    text = RE_DOLLAR_XXXX.sub("[AMOUNT]", text)
    
    # 3. Normalize generic names / account number redactions
    text = RE_GENERIC_XXXX.sub("[REDACTED]", text)
    
    return text


def clean_narrative(text: Optional[str]) -> str:
    """
    Full cleaning pipeline for a consumer complaint narrative.
    
    Args:
        text: Raw narrative text from CFPB record
        
    Returns:
        Cleaned, normalized, single-spaced string ready for embeddings.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Normalize unicode characters
    cleaned = normalize_unicode(text)
    
    # Step 2: Normalize PII redaction tokens
    cleaned = normalize_pii_redactions(cleaned)
    
    # Step 3: Clean horizontal whitespace and linebreaks cleanly
    cleaned = RE_MULTIPLE_WHITESPACE.sub(" ", cleaned)
    cleaned = RE_WHITESPACE_NEWLINES.sub("\n", cleaned)
    cleaned = RE_MULTIPLE_NEWLINES.sub("\n\n", cleaned)
    
    return cleaned.strip()


def is_valid_narrative(text: Optional[str], min_words: int = 15) -> bool:
    """
    Validates whether a narrative contains enough meaningful content for RAG indexing.
    Rejects empty, boilerplate, or extremely short non-substantive text.
    
    Args:
        text: Complaint narrative text
        min_words: Minimum word threshold (default: 15)
        
    Returns:
        True if narrative meets quality threshold, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False
    
    clean = text.strip()
    words = clean.split()
    
    if len(words) < min_words:
        return False
        
    # Check for useless boilerplate phrases
    boilerplate_phrases = [
        "see attached",
        "please see attached",
        "refer to attached document",
        "n/a",
        "none",
    ]
    
    if clean.lower() in boilerplate_phrases:
        return False
        
    return True

