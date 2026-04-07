from __future__ import annotations

import unittest

from services.legal_ai_service.application.services.legal_ai_scaffold_service import (
    LegalAiScaffoldService,
)


class LegalAiScaffoldServiceTests(unittest.TestCase):
    def test_build_ping_response_exposes_reserved_routes(self) -> None:
        response = LegalAiScaffoldService().build_ping_response()
        self.assertEqual(response.service_name, "legal_ai_service")
        self.assertIn("/legal-ai/ping", response.reserved_routes)


if __name__ == "__main__":
    unittest.main()
