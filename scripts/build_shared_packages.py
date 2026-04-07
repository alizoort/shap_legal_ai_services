#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_PATH = REPO_ROOT / "shared_packages" / "versions.toml"

PACKAGE_SPECS = {
    "shap-contracts": {
        "import_name": "shap_contracts",
        "source_root": REPO_ROOT / "shap_contracts",
        "dependencies": ["pydantic>=2,<3"],
        "description": "Versioned shared contracts for SHAP Legal AI services.",
    },
    "shap-service-runtime": {
        "import_name": "shap_service_runtime",
        "source_root": REPO_ROOT / "shap_service_runtime",
        "dependencies": ["shap-contracts>=0.1.0,<1.0.0"],
        "optional_dependencies": {
            "auth": ["fastapi>=0.115,<1"],
            "db": ["alembic>=1.13,<2", "sqlalchemy>=2,<3"],
            "messaging": ["kombu>=5.3,<6"],
            "web": ["fastapi>=0.115,<1", "prometheus-client>=0.20,<1", "sqlalchemy>=2,<3"],
            "all": [
                "alembic>=1.13,<2",
                "fastapi>=0.115,<1",
                "kombu>=5.3,<6",
                "prometheus-client>=0.20,<1",
                "sqlalchemy>=2,<3",
            ],
        },
        "description": "Versioned shared runtime helpers for SHAP Legal AI services.",
    },
}


def load_versions() -> dict[str, str]:
    data = tomllib.loads(VERSIONS_PATH.read_text(encoding="utf-8"))
    packages = data.get("packages", {})
    return {
        name: str(packages[name]["version"])
        for name in PACKAGE_SPECS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage and optionally build SHAP shared packages.")
    parser.add_argument("--output-dir", default="dist/shared-packages")
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def write_pyproject(stage_root: Path, package_name: str, version: str) -> None:
    spec = PACKAGE_SPECS[package_name]
    import_name = spec["import_name"]
    dependencies = ",\n    ".join(f'"{item}"' for item in spec.get("dependencies", []))
    optional_blocks: list[str] = []
    for extra_name, extra_dependencies in spec.get("optional_dependencies", {}).items():
        rendered = ",\n        ".join(f'"{item}"' for item in extra_dependencies)
        optional_blocks.append(f'"{extra_name}" = [\n        {rendered}\n    ]')

    pyproject = f"""[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
version = "{version}"
description = "{spec["description"]}"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    {dependencies}
]
"""
    if optional_blocks:
        pyproject += "\n[project.optional-dependencies]\n" + "\n".join(optional_blocks) + "\n"

    pyproject += f"""
[tool.setuptools]
include-package-data = true

[tool.setuptools.packages.find]
include = ["{import_name}", "{import_name}.*"]
"""
    (stage_root / "pyproject.toml").write_text(pyproject.strip() + "\n", encoding="utf-8")


def stage_package(package_name: str, version: str, output_root: Path) -> None:
    spec = PACKAGE_SPECS[package_name]
    stage_root = output_root / "staging" / package_name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    destination = stage_root / spec["import_name"]
    shutil.copytree(spec["source_root"], destination, dirs_exist_ok=True)
    (stage_root / "README.md").write_text(
        f"# {package_name}\n\nPublished shared package extracted from the SHAP Legal AI services repository.\n",
        encoding="utf-8",
    )
    write_pyproject(stage_root, package_name, version)

    if output_root.joinpath("dist").exists():
        output_root.joinpath("dist", package_name).mkdir(parents=True, exist_ok=True)


def build_package(package_name: str, output_root: Path) -> None:
    stage_root = output_root / "staging" / package_name
    dist_root = output_root / "dist" / package_name
    dist_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", str(dist_root)],
        check=True,
        cwd=stage_root,
    )


def main() -> int:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    versions = load_versions()
    for package_name, version in versions.items():
        stage_package(package_name, version, output_root)
        if not args.no_build:
            build_package(package_name, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
