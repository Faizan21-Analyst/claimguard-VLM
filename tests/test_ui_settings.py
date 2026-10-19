from claimguard.ui.settings import UiSettings


def test_defaults(monkeypatch):
    monkeypatch.delenv("UI_API_URL", raising=False)
    monkeypatch.delenv("UI_TIMEOUT_S", raising=False)

    settings = UiSettings(_env_file=None)

    assert settings.api_url == "http://localhost:8000"
    assert settings.timeout_s == 120.0


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("UI_API_URL", "http://api:9000")
    monkeypatch.setenv("UI_TIMEOUT_S", "30")

    settings = UiSettings(_env_file=None)

    assert settings.api_url == "http://api:9000"
    assert settings.timeout_s == 30.0


def test_ui_settings_never_expose_the_groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-key-123")

    settings = UiSettings(_env_file=None)

    assert "secret-key-123" not in settings.model_dump_json()
    assert not hasattr(settings, "groq")