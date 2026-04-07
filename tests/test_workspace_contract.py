from __future__ import annotations

from pathlib import Path
import unittest


class WorkspaceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.workspace_root = cls.repo_root.parent

    def test_required_backend_paths_exist(self) -> None:
        for relative_path in (
            "docs/architecture/backend-system-instructions.md",
            "docs/architecture/manifests/legal-ai-scaffold.manifest.json",
            "docs/architecture/manifests/legal-ai-scaffold.manifest.instructions.md",
            "shap_service_runtime",
            "shap_contracts",
            "services/gateway_service",
            "services/legal_ai_service",
        ):
            self.assertTrue((self.repo_root / relative_path).exists(), relative_path)

    def test_workspace_bootstrap_artifacts_exist(self) -> None:
        for relative_path in (
            "AGENTS.md",
            "docs/README.md",
            "shap_legal_ai_web",
            "shap_legal_ai_services",
        ):
            self.assertTrue((self.workspace_root / relative_path).exists(), relative_path)

    def test_workspace_has_no_leftover_reference_names(self) -> None:
        ignored_suffixes = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".jar",
        }
        banned_terms = ("".join(["si", "ren"]), "".join(["lm", "s"]))
        for root_name in ("shap_legal_ai_web", "shap_legal_ai_services"):
            for path in (self.workspace_root / root_name).rglob("*"):
                if not path.is_file() or path.suffix.lower() in ignored_suffixes:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                lowered = text.lower()
                for term in banned_terms:
                    self.assertNotIn(term, lowered, str(path))


if __name__ == "__main__":
    unittest.main()
