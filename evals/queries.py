"""
Phase 4 golden evaluation queries.

50 queries: 40 stratified across the 5 target banks x 4 product families
(2 issues per company x product cell), phrased as a compliance officer
would actually ask them — plus 10 cross-bank queries with no company filter,
to test retrieval on more open-ended questions.

Company and product-family names are pulled from settings, not hardcoded,
so this file can never drift out of sync with what the corpus actually
contains (see src/config.py TARGET_COMPANIES / TARGET_PRODUCT_FAMILIES).
"""

from typing import Dict, List, Optional, TypedDict

from src.config import settings


class EvalQuery(TypedDict):
    query_id: str
    query: str
    company: Optional[str]          # None for cross-bank queries
    product_family: Optional[str]


# Realistic display names for natural-language phrasing — the raw legal
# entity names in settings.TARGET_COMPANIES ("JPMORGAN CHASE & CO.") read
# like a legal filing, not a question a person would actually type.
_DISPLAY_NAME: Dict[str, str] = {
    "WELLS FARGO & COMPANY": "Wells Fargo",
    "JPMORGAN CHASE & CO.": "Chase",
    "BANK OF AMERICA, NATIONAL ASSOCIATION": "Bank of America",
    "CAPITAL ONE FINANCIAL CORPORATION": "Capital One",
    "CITIBANK, N.A.": "Citibank",
}

# Two realistic issue phrasings per product family, based on patterns
# actually observed in the real corpus during Phase 1-3 spot checks.
_ISSUE_TEMPLATES: Dict[str, List[str]] = {
    "Checking or savings account": [
        "an unauthorized transaction on a checking account that the bank refused to refund",
        "overdraft fees charged without proper notice",
    ],
    "Credit card": [
        "an annual fee charged on a card advertised as having no fees",
        "a fraudulent charge the cardholder disputed but was never credited",
    ],
    "Mortgage": [
        "a mortgage servicer misapplying monthly payments",
        "an escrow account shortage the bank failed to explain",
    ],
    "Debt collection": [
        "aggressive debt collection calls after the debt was already paid",
        "a debt collector reporting inaccurate information to credit bureaus",
    ],
}

# 10 cross-bank questions — no target company, so retrieval has to work on
# meaning/topic alone rather than a company name doing most of the work.
_CROSS_BANK_QUERIES: List[tuple] = [
    ("unauthorized wire transfers that banks refused to reverse", "Checking or savings account"),
    ("credit card annual fees charged despite no-fee promises", "Credit card"),
    ("mortgage escrow shortages across multiple lenders", "Mortgage"),
    ("aggressive debt collection tactics after debts were already settled", "Debt collection"),
    ("overdraft fee disputes resolved with monetary relief", "Checking or savings account"),
    ("credit card fraud disputes that were denied without investigation", "Credit card"),
    ("mortgage payment misapplication causing incorrect late fees", "Mortgage"),
    ("debt collectors reporting inaccurate information to credit bureaus", "Debt collection"),
    ("checking account closures without adequate notice to customers", "Checking or savings account"),
    ("disputes over unexpected credit card interest rate increases", "Credit card"),
]


def _build_eval_queries() -> List[EvalQuery]:
    queries: List[EvalQuery] = []
    qid = 1

    for company in settings.TARGET_COMPANIES:
        for product_family, issues in _ISSUE_TEMPLATES.items():
            for issue in issues:
                queries.append({
                    "query_id": f"q{qid:03d}",
                    "query": f"{_DISPLAY_NAME[company]} complaints about {issue}",
                    "company": company,
                    "product_family": product_family,
                })
                qid += 1

    for issue, product_family in _CROSS_BANK_QUERIES:
        queries.append({
            "query_id": f"q{qid:03d}",
            "query": issue,
            "company": None,
            "product_family": product_family,
        })
        qid += 1

    return queries


EVAL_QUERIES: List[EvalQuery] = _build_eval_queries()
