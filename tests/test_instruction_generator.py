from __future__ import annotations

import unittest

from scripts.instructions.generate_instructions import generate_instruction_markdown, validate_manifest


class InstructionGeneratorTests(unittest.TestCase):
    def test_generator_accepts_scaffold_manifest(self) -> None:
        manifest = {
            "title": "Legal AI Scaffold",
            "scope": "service",
            "bounded_context": "legal_ai",
            "use_cases": ["Reserve the legal-ai gateway and service seams."],
            "commands": ["Refresh placeholder legal-ai status."],
            "queries": ["Read scaffold readiness."],
            "events_emitted": ["PlaceholderReserved"],
            "events_consumed": [],
            "api_endpoints": [
                {"method": "GET", "path": "/legal-ai/ping", "purpose": "Scaffold health handshake"}
            ],
            "data_ownership": {
                "service_database": "legal_ai_db",
                "owned_entities": ["placeholder scaffold state"],
                "cross_service_access": "No direct cross-service database access.",
            },
            "dependencies": ["gateway_service", "shap_service_runtime", "shap_contracts"],
            "security_constraints": ["Gateway remains the user-context boundary seam."],
            "observability": {"logs": ["startup"], "metrics": ["http"], "alerts": []},
            "test_plan": {
                "unit": ["Scaffold service response"],
                "contract": ["Gateway reserved route"],
                "integration": ["Compose boot"],
                "acceptance": ["Reserved route reachable"],
            },
        }

        validate_manifest(manifest)
        markdown = generate_instruction_markdown(manifest)
        self.assertIn("Legal AI Scaffold", markdown)


if __name__ == "__main__":
    unittest.main()
