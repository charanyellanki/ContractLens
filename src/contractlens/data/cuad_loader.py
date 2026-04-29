"""CUAD dataset loader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json


@dataclass
class CUADContract:
    """A contract from the CUAD dataset."""

    contract_id: str
    text: str
    clauses: dict[str, list[dict]]


class CUADLoader:
    """Loader for the CUAD dataset."""

    def __init__(self, data_path: Optional[str] = None) -> None:
        self.data_path = Path(data_path) if data_path else Path("./data/cuad")

    def load_contract(self, contract_id: str) -> CUADContract:
        """Load a single contract by ID."""
        # TODO: Implement actual CUAD loading
        # This is a stub that returns sample data
        return CUADContract(
            contract_id=contract_id,
            text="Sample contract text...",
            clauses={},
        )

    def load_train(self) -> list[CUADContract]:
        """Load training contracts."""
        return []

    def load_test(self) -> list[CUADContract]:
        """Load test contracts."""
        return []

    def load_categories(self) -> list[str]:
        """Load the 41 CUAD clause categories."""
        return [
            "Confidentiality",
            "Non-Disclosure",
            "Non-Disclosure Agreement",
            "IP Ownership",
            "License Grant",
            "Patent Rights",
            "Copyright",
            "Trademark",
            "Limitation of Liability",
            "Indemnification",
            "Indemnification Cap",
            "Mutual Indemnification",
            "Termination",
            "Termination for Convenience",
            "Termination for Cause",
            "Survival",
            "Payment Terms",
            "Price and Payment",
            "Taxes",
            "Expenses",
            "Invoicing",
            "Warranty",
            "Warranties",
            "Representations",
            "Disclaimer",
            "Force Majeure",
            "Governing Law",
            "Jurisdiction",
            "Venue",
            "Assignment",
            "Amendment",
            "Waiver",
            "Notices",
            "Entire Agreement",
            "Severability",
            "Relationship",
            "Compliance",
            "Regulatory",
            "Export Control",
            "Anti-Corruption",
            "Non-Compete",
            "Non-Solicitation",
            "Insurance",
            "Security",
            "Data Protection",
            "Privacy",
        ]