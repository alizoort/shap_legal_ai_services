from __future__ import annotations

import os
import unittest

os.environ.setdefault("ALLOW_DIRECT_PACKAGE_IMPORTS", "true")

_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from fastapi.testclient import TestClient
    from services.legal_ai_service.app.main import create_app
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
    TestClient = None
    create_app = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing dependency: {_IMPORT_ERROR}")
class LegalAiControllerTests(unittest.TestCase):
    def test_ping_endpoint_returns_placeholder_payload(self) -> None:
        assert TestClient is not None
        assert create_app is not None
        client = TestClient(create_app())

        response = client.get("/legal-ai/ping")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service_name"], "legal_ai_service")


if __name__ == "__main__":
    unittest.main()
