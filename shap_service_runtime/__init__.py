"""Shared runtime helpers used across SHAP Legal AI services."""

import os
import sys

# Phase 2: Prevent direct imports - enforce package installation
# This ensures services install packages from artifact repository
if os.getenv("ALLOW_DIRECT_PACKAGE_IMPORTS", "").lower() != "true":
    # Check if module is imported from installed package location
    if "site-packages" not in __file__ and "dist-packages" not in __file__:
        raise ImportError(
            "shap_service_runtime must be installed as a package.\n"
            "\n"
            "Install from artifact repository:\n"
            "  pip install shap-service-runtime\n"
            "\n"
            "Or for local development, set environment variable:\n"
            "  ALLOW_DIRECT_PACKAGE_IMPORTS=true\n"
            "\n"
            "This requirement ensures services work independently when separated into different repositories."
        )

__all__ = [
    "auth",
    "config",
    "db",
    "messaging",
    "web",
]
