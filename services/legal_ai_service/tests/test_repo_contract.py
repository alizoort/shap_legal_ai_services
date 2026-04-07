from __future__ import annotations

from pathlib import Path
import unittest


class LegalAiServiceRepoContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service_root = Path(__file__).resolve().parents[1]

    def test_repo_ready_artifacts_exist(self) -> None:
        for relative_path in (
            "Dockerfile",
            "README.md",
            "pyproject.toml",
            "requirements.txt",
            "run_migrations.py",
            "services/__init__.py",
            "services/legal_ai_service/__init__.py",
            "tests/__init__.py",
            "tests/test_repo_contract.py",
            "tests/test_legal_ai_scaffold_service.py",
            "tests/test_legal_ai_controller.py",
        ):
            self.assertTrue((self.service_root / relative_path).exists(), relative_path)

    def test_readme_documents_local_quality_gate_commands(self) -> None:
        readme_text = (self.service_root / "README.md").read_text(encoding="utf-8")
        self.assertIn("ruff check .", readme_text)
        self.assertIn("mypy .", readme_text)
        self.assertIn("python3 -m unittest discover -s tests", readme_text)


if __name__ == "__main__":
    unittest.main()
