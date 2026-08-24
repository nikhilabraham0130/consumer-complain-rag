"""
Unit tests for PII normalization, text sanitization, and narrative validation logic.
"""

import pytest
from src.ingestion.cleaner import (
    clean_narrative,
    is_valid_narrative,
    normalize_pii_redactions,
    normalize_unicode,
)


def test_date_redaction_normalization() -> None:
    """Tests that various CFPB date redaction formats are replaced with [DATE]."""
    text_1 = "On XX/XX/2023 I noticed a suspicious transaction."
    text_2 = "On XX/XX/XXXX my account was opened without permission."
    text_3 = "Transaction date: 11/XX/2022."
    
    assert normalize_pii_redactions(text_1) == "On [DATE] I noticed a suspicious transaction."
    assert normalize_pii_redactions(text_2) == "On [DATE] my account was opened without permission."
    assert normalize_pii_redactions(text_3) == "Transaction date: [DATE]."


def test_amount_redaction_normalization() -> None:
    """Tests that currency redaction formats ({$100.00}, {$XX.XX}) are replaced with [AMOUNT]."""
    text_1 = "They charged me {$450.00} for overdraft fees."
    text_2 = "An unauthorized charge of {$XX.XX} occurred."
    text_3 = "I paid $XXXX to the collection agency."
    
    assert normalize_pii_redactions(text_1) == "They charged me [AMOUNT] for overdraft fees."
    assert normalize_pii_redactions(text_2) == "An unauthorized charge of [AMOUNT] occurred."
    assert normalize_pii_redactions(text_3) == "I paid [AMOUNT] to the collection agency."


def test_generic_xxxx_redaction_normalization() -> None:
    """Tests that company/name/account XXXX sequences are replaced with [REDACTED]."""
    text = "I spoke with XXXX at XXXX branch regarding my account."
    assert normalize_pii_redactions(text) == "I spoke with [REDACTED] at [REDACTED] branch regarding my account."


def test_unicode_normalization() -> None:
    """Tests replacing curly quotes, non-breaking spaces, and em dashes."""
    text = "It\u2019s my money\u2014I didn\u2019t authorize this\u00a0transfer."
    normalized = normalize_unicode(text)
    assert normalized == "It's my money - I didn't authorize this transfer."


def test_full_clean_narrative_pipeline() -> None:
    """Tests end-to-end cleaning pipeline on a realistic messy complaint."""
    raw = (
        "On  XX/XX/2023,   I noticed an unauthorized charge of   {$350.00}   \n\n\n\n"
        "from XXXX on my account. I spoke with representative XXXX.\n\n"
        "They refused to reverse it."
    )
    cleaned = clean_narrative(raw)
    
    expected = (
        "On [DATE], I noticed an unauthorized charge of [AMOUNT]\n\n"
        "from [REDACTED] on my account. I spoke with representative [REDACTED].\n\n"
        "They refused to reverse it."
    )
    assert cleaned == expected


def test_is_valid_narrative_filter() -> None:
    """Tests narrative quality filter rejecting empty, short, or boilerplate text."""
    # Valid long narrative
    valid_text = "On [DATE] I discovered that an unauthorized transfer was made from my account and the bank refused to refund it."
    assert is_valid_narrative(valid_text, min_words=15) is True
    
    # Too short (under 15 words)
    too_short = "Bank took my money. Bad service."
    assert is_valid_narrative(too_short, min_words=15) is False
    
    # Boilerplate junk
    assert is_valid_narrative("Please see attached", min_words=15) is False
    assert is_valid_narrative("N/A", min_words=15) is False
    assert is_valid_narrative("", min_words=15) is False
    assert is_valid_narrative(None, min_words=15) is False

