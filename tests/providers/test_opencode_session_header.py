"""Tests for OpenCode session header injection in OpenAICompatibleProvider."""

import re
import uuid
from unittest.mock import patch

import httpx

from kader.providers.openai_compatible import (
    OPENCODE_SESSION_PROVIDERS,
    OpenAICompatibleProvider,
    OpenAIProviderConfig,
    _generate_session_id,
)


class TestGenerateSessionId:
    def test_format(self):
        sid = _generate_session_id()
        assert sid.startswith("ses_")
        # remainder is a valid uuid hex
        uuid.UUID(sid[4:])

    def test_unique(self):
        assert _generate_session_id() != _generate_session_id()


class TestOpenCodeSessionHeaders:
    def test_opencode_provider_gets_session_header(self):
        provider = OpenAICompatibleProvider(
            model="claude-sonnet-4-5",
            provider_config=OpenAIProviderConfig(
                api_key="test-key",
                base_url="https://opencode.ai/zen/v1",
            ),
        )
        headers = provider._client.default_headers
        assert headers["x-opencode-session"].startswith("ses_")
        assert headers["x-opencode-client"] == "kader"

    def test_opencode_go_provider_gets_session_header(self):
        provider = OpenAICompatibleProvider(
            model="glm-5.1",
            provider_config=OpenAIProviderConfig(
                api_key="test-key",
                base_url="https://opencode.ai/zen/go/v1",
            ),
        )
        headers = provider._client.default_headers
        assert headers["x-opencode-session"].startswith("ses_")
        assert headers["x-opencode-client"] == "kader"

    def test_async_client_also_gets_header(self):
        provider = OpenAICompatibleProvider(
            model="gpt-4o",
            provider_config=OpenAIProviderConfig(
                api_key="test-key",
                base_url="https://opencode.ai/zen/v1",
            ),
        )
        headers = provider._async_client.default_headers
        assert (
            headers["x-opencode-session"]
            == provider._client.default_headers["x-opencode-session"]
        )

    def test_session_id_stable_per_instance(self):
        """Session id is generated once per provider instance, not per request."""
        provider = OpenAICompatibleProvider(
            model="gpt-4o",
            provider_config=OpenAIProviderConfig(
                api_key="test-key",
                base_url="https://opencode.ai/zen/v1",
            ),
        )
        sid = provider._client.default_headers["x-opencode-session"]
        assert provider._async_client.default_headers["x-opencode-session"] == sid

    def test_user_headers_take_precedence(self):
        provider = OpenAICompatibleProvider(
            model="gpt-4o",
            provider_config=OpenAIProviderConfig(
                api_key="test-key",
                base_url="https://opencode.ai/zen/v1",
                default_headers={"x-opencode-session": "ses_custom"},
            ),
        )
        headers = provider._client.default_headers
        assert headers["x-opencode-session"] == "ses_custom"

    def test_non_opencode_provider_no_session_header(self):
        provider = OpenAICompatibleProvider(
            model="gpt-4o",
            provider_config=OpenAIProviderConfig(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            ),
        )
        headers = provider._client.default_headers
        assert "x-opencode-session" not in headers
        assert "x-opencode-client" not in headers

    def test_session_providers_constant(self):
        assert OPENCODE_SESSION_PROVIDERS == ("opencode", "opencode_go")


class TestSessionHeaderSentOnRequest:
    def test_header_present_on_http_request(self):
        """Verify the header is actually attached at the HTTP layer."""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["session"] = request.headers.get("x-opencode-session")
            captured["client"] = request.headers.get("x-opencode-client")
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "claude-sonnet-4-5",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        provider = OpenAICompatibleProvider(
            model="claude-sonnet-4-5",
            provider_config=OpenAIProviderConfig(
                api_key="test-key",
                base_url="https://opencode.ai/zen/v1",
            ),
        )
        # Swap in a mock transport while keeping session default headers
        # (filter out openai SDK sentinel values like Omit)
        clean_headers = {
            k: v
            for k, v in provider._client.default_headers.items()
            if isinstance(v, str)
        }
        with patch.object(
            provider._client,
            "_client",
            httpx.Client(
                transport=httpx.MockTransport(handler),
                headers=clean_headers,
            ),
        ):
            provider.invoke([])

        assert captured["session"] and re.match(
            r"^ses_[0-9a-f]{32}$", captured["session"]
        )
        assert captured["client"] == "kader"
