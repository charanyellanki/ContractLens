"""Extraction prompts for each CUAD clause category."""

from contractlens.models import ClauseCategory


# Base prompt template for clause extraction
EXTRACTION_PROMPT_TEMPLATE = """You are a legal contract analysis expert. Your task is to extract {category} clauses from the given contract text.

For each {category} clause found, extract:
1. The exact text of the clause
2. Character offsets (start_char, end_char) referencing the original text
3. A confidence score (0.0-1.0) indicating how certain you are this is a {category} clause

Return your response as a JSON array of objects with this structure:
[
  {{
    "text": "exact clause text here",
    "start_char": 1234,
    "end_char": 1567,
    "confidence": 0.95
  }}
]

If no {category} clauses are found, return an empty array: []

Contract text:
{contract_text}

Context (from contract retrieval):
{context}
"""


# Category-specific prompts
EXTRACTION_PROMPTS: dict[ClauseCategory, str] = {
    ClauseCategory.CONFIDENTIALITY: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Confidentiality",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.NON_DISCLOSURE: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Non-Disclosure",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.NON_DISCLOSURE_AGREEMENT: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Non-Disclosure Agreement",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.IP_OWNERSHIP: EXTRACTION_PROMPT_TEMPLATE.format(
        category="IP Ownership",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.LICENSE_GRANT: EXTRACTION_PROMPT_TEMPLATE.format(
        category="License Grant",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.PATENT_RIGHTS: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Patent Rights",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.COPYRIGHT: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Copyright",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.TRADEMARK: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Trademark",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.LIMITATION_OF_LIABILITY: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Limitation of Liability",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.INDEMNIFICATION: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Indemnification",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.INDEMNIFICATION_CAP: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Indemnification Cap",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.MUTUAL_INDEMNIFICATION: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Mutual Indemnification",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.TERMINATION: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Termination",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.TERMINATION_FOR_CONVENIENCE: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Termination for Convenience",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.TERMINATION_FOR_CAUSE: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Termination for Cause",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.SURVIVAL: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Survival",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.PAYMENT_TERMS: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Payment Terms",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.PRICE_AND_PAYMENT: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Price and Payment",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.TAXES: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Taxes",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.EXPENSES: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Expenses",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.INVOICING: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Invoicing",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.WARRANTY: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Warranty",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.WARRANTIES: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Warranties",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.REPRESENTATIONS: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Representations",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.DISCLAIMER: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Disclaimer",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.FORCE_MAJEURE: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Force Majeure",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.GOVERNING_LAW: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Governing Law",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.JURISDICTION: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Jurisdiction",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.VENUE: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Venue",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.ASSIGNMENT: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Assignment",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.AMENDMENT: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Amendment",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.WAIVER: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Waiver",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.NOTICES: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Notices",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.ENTIRE_AGREEMENT: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Entire Agreement",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.SEVERABILITY: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Severability",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.RELATIONSHIP: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Relationship",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.COMPLIANCE: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Compliance",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.REGULATORY: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Regulatory",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.EXPORT_CONTROL: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Export Control",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.ANTI_CORRUPTION: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Anti-Corruption",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.NON_COMPETE: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Non-Compete",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.NON_SOLICITATION: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Non-Solicitation",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.INSURANCE: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Insurance",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.SECURITY: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Security",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.DATA_PROTECTION: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Data Protection",
        contract_text="{contract_text}",
        context="{context}",
    ),
    ClauseCategory.PRIVACY: EXTRACTION_PROMPT_TEMPLATE.format(
        category="Privacy",
        contract_text="{contract_text}",
        context="{context}",
    ),
}


def get_extraction_prompt(category: ClauseCategory) -> str:
    """Get the extraction prompt for a specific category."""
    return EXTRACTION_PROMPTS.get(category, EXTRACTION_PROMPT_TEMPLATE.format(
        category=category.value,
        contract_text="{contract_text}",
        context="{context}",
    ))