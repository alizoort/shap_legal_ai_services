from __future__ import annotations

import os
import unittest

os.environ.setdefault("ALLOW_DIRECT_PACKAGE_IMPORTS", "true")

_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from services.gateway_service.app.proxy import build_upstream_url, sanitize_request_headers
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
    build_upstream_url = None
    sanitize_request_headers = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing dependency: {_IMPORT_ERROR}")
class GatewayProxyTests(unittest.TestCase):
    def test_build_upstream_url_preserves_query_string(self) -> None:
        assert build_upstream_url is not None
        result = build_upstream_url(
            upstream_base_url="http://legal-ai-service:8103",
            path="/legal-ai/ping",
            query_string="verbose=true",
        )
        self.assertEqual(result, "http://legal-ai-service:8103/legal-ai/ping?verbose=true")

    def test_sanitize_request_headers_replaces_signed_headers(self) -> None:
        assert sanitize_request_headers is not None
        result = sanitize_request_headers(
            {
                "Authorization": "Bearer ignored",
                "X-User-Id": "upstream",
                "Accept": "application/json",
            },
            extra_headers={"X-User-Id": "anonymous-dev"},
        )
        self.assertEqual(result["X-User-Id"], "anonymous-dev")
        self.assertNotIn("Authorization", result)


if __name__ == "__main__":
    unittest.main()
