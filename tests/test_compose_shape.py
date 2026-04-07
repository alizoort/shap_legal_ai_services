from __future__ import annotations

from pathlib import Path
import unittest


class ComposeShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_dev_compose_contains_expected_services(self) -> None:
        compose_text = (self.repo_root / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn('app:', compose_text)
        self.assertIn('legal-ai-service:', compose_text)
        self.assertIn('"8000:8000"', compose_text)
        self.assertIn('"8103:8103"', compose_text)
        self.assertNotIn('rabbitmq:', compose_text)
        self.assertNotIn('redis:', compose_text)
        self.assertNotIn('keycloak:', compose_text)


if __name__ == "__main__":
    unittest.main()
