"""Tests for Kader CLI settings module."""

import json
from pathlib import Path

import pytest

from cli.settings.settings import (
    KaderSettings,
    load_settings,
    save_settings,
)

# ── KaderSettings dataclass ──────────────────────────────────────────


class TestKaderSettingsDefaults:
    """Test that default values are correct."""

    def test_default_values(self) -> None:
        settings = KaderSettings()
        assert settings.main_agent_provider == "ollama"
        assert settings.sub_agent_provider == "ollama"
        assert settings.main_agent_model == "glm-5:cloud"
        assert settings.sub_agent_model == "glm-5:cloud"


class TestKaderSettingsValidation:
    """Test provider validation logic."""

    def test_valid_provider(self) -> None:
        settings = KaderSettings(main_agent_provider="google")
        assert settings.main_agent_provider == "google"

    def test_invalid_main_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid provider"):
            KaderSettings(main_agent_provider="nonexistent")

    def test_invalid_sub_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid provider"):
            KaderSettings(sub_agent_provider="bad_provider")

    def test_all_known_providers_accepted(self) -> None:
        for provider in KaderSettings.VALID_PROVIDERS:
            settings = KaderSettings(
                main_agent_provider=provider, sub_agent_provider=provider
            )
            assert settings.main_agent_provider == provider


# ── to_dict / from_dict ──────────────────────────────────────────────


class TestKaderSettingsSerialisation:
    """Test JSON key mapping round-trip."""

    def test_to_dict_uses_hyphenated_keys(self) -> None:
        settings = KaderSettings()
        d = settings.to_dict()
        assert "main-agent-provider" in d
        assert "sub-agent-model" in d
        assert d["main-agent-provider"] == "ollama"

    def test_from_dict_with_hyphenated_keys(self) -> None:
        data = {
            "main-agent-provider": "google",
            "sub-agent-provider": "mistral",
            "main-agent-model": "gemini-flash",
            "sub-agent-model": "mistral-sm",
        }
        settings = KaderSettings.from_dict(data)
        assert settings.main_agent_provider == "google"
        assert settings.sub_agent_provider == "mistral"
        assert settings.main_agent_model == "gemini-flash"
        assert settings.sub_agent_model == "mistral-sm"

    def test_from_dict_missing_keys_use_defaults(self) -> None:
        settings = KaderSettings.from_dict({"main-agent-provider": "anthropic"})
        assert settings.main_agent_provider == "anthropic"
        assert settings.sub_agent_provider == "ollama"  # default

    def test_from_dict_ignores_unknown_keys(self) -> None:
        settings = KaderSettings.from_dict({"unknown-key": "value"})
        assert settings.main_agent_provider == "ollama"

    def test_round_trip(self) -> None:
        original = KaderSettings(
            main_agent_provider="google",
            sub_agent_provider="anthropic",
            main_agent_model="gemini-pro",
            sub_agent_model="claude-3",
        )
        restored = KaderSettings.from_dict(original.to_dict())
        assert restored.main_agent_provider == original.main_agent_provider
        assert restored.sub_agent_provider == original.sub_agent_provider
        assert restored.main_agent_model == original.main_agent_model
        assert restored.sub_agent_model == original.sub_agent_model


# ── Model string helpers ─────────────────────────────────────────────


class TestModelStringHelpers:
    """Test get_main_model_string / get_sub_model_string."""

    def test_main_model_string(self) -> None:
        settings = KaderSettings(
            main_agent_provider="google", main_agent_model="gemini-flash"
        )
        assert settings.get_main_model_string() == "google:gemini-flash"

    def test_sub_model_string_default(self) -> None:
        settings = KaderSettings()
        assert settings.get_sub_model_string() == "ollama:glm-5:cloud"


# ── load_settings / save_settings ─────────────────────────────────────


class TestLoadSaveSettings:
    """Test file I/O round-trip."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        original = KaderSettings(
            main_agent_provider="google",
            main_agent_model="gemini-pro",
        )
        save_settings(original, path)

        loaded = load_settings(path)
        assert loaded.main_agent_provider == "google"
        assert loaded.main_agent_model == "gemini-pro"

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "no_such_file.json"
        settings = load_settings(path)
        assert settings.main_agent_provider == "ollama"
        assert settings.main_agent_model == "glm-5:cloud"

    def test_load_corrupt_json_returns_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text("{invalid json!!", encoding="utf-8")
        settings = load_settings(path)
        assert settings.main_agent_provider == "ollama"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "settings.json"
        save_settings(KaderSettings(), path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "main-agent-provider" in data

    def test_saved_json_format(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        save_settings(KaderSettings(), path)
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        expected_keys = {
            "main-agent-provider",
            "sub-agent-provider",
            "main-agent-model",
            "sub-agent-model",
            "auto-update",
        }
        assert set(data.keys()) == expected_keys
