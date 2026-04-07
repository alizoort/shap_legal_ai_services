#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import textwrap
import venv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate built SHAP shared packages in an isolated virtualenv.")
    parser.add_argument("--dist-root", required=True)
    parser.add_argument("--test-requirements", default="shared_packages/test_requirements.txt")
    return parser.parse_args()


def build_virtualenv(venv_root: Path) -> tuple[Path, Path]:
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_root)
    bin_dir = venv_root / ("Scripts" if sys.platform.startswith("win") else "bin")
    return bin_dir / "python", bin_dir / "pip"


def run(argv: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(argv, check=True, cwd=cwd)


def main() -> int:
    args = parse_args()
    dist_root = Path(args.dist_root).resolve()
    test_requirements = Path(args.test_requirements).resolve()
    distribution_files = sorted(dist_root.glob("*/*"))
    if not distribution_files:
        raise FileNotFoundError(f"No built distributions found under {dist_root}")

    with tempfile.TemporaryDirectory(prefix="shap-shared-packages-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        python, pip = build_virtualenv(tmp_root / "venv")

        run([str(pip), "install", "--upgrade", "pip"])
        run([str(pip), "install", *[str(path) for path in distribution_files]])
        if test_requirements.exists():
            run([str(pip), "install", "-r", str(test_requirements)])

        smoke_script = tmp_root / "artifact_smoke.py"
        smoke_script.write_text(
            textwrap.dedent(
                """
                from shap_contracts.events import EventEnvelope, PlaceholderReservedPayload
                from shap_service_runtime.auth.context_signer import UserContextSigner
                from shap_service_runtime.auth.models import SignedUserContext
                from shap_service_runtime.db.alembic import MigrationTarget

                payload = PlaceholderReservedPayload(
                    placeholder_key="legal_ai_scaffold",
                    detail="reserved",
                )
                envelope = EventEnvelope[PlaceholderReservedPayload](
                    event_type="PlaceholderReserved",
                    producer="artifact-smoke",
                    payload=payload,
                )
                assert envelope.payload.placeholder_key == "legal_ai_scaffold"

                signer = UserContextSigner(secret="artifact-secret", max_age_seconds=60)
                signed = signer.build_signed_headers(
                    SignedUserContext(
                        user_id="user-1",
                        roles=["USER"],
                        permissions=["APP_ACCESS"],
                        email="user@example.com",
                    )
                )
                verified = signer.verify(
                    user_id=signed.user_id,
                    roles=signed.roles,
                    permissions=signed.permissions,
                    email=signed.email,
                    issued_at=signed.issued_at,
                    signature=signed.signature,
                )
                assert verified.user_id == "user-1"
                assert MigrationTarget(
                    service_name="legal_ai_service",
                    alembic_config_path="/tmp/alembic.ini",
                    database_url="postgresql://db/legal_ai_db",
                ).service_name == "legal_ai_service"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        run([str(python), str(smoke_script)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
