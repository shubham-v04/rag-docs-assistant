import pytest

from app.config import Settings

MODEL_VARS = ["GENERATION_MODEL", "GRADER_MODEL", "GROUNDING_MODEL"]


@pytest.fixture
def clean_env(monkeypatch):
    for name in MODEL_VARS:
        # set then delete, so monkeypatch removes anything a test sets afterwards
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_every_model_defaults_to_the_tested_one(clean_env):
    settings = Settings(_env_file=None)

    assert settings.generation_model == "openai/gpt-oss-20b"
    assert settings.grader_model == "openai/gpt-oss-20b"
    assert settings.grounding_model == "openai/gpt-oss-20b"


def test_each_model_can_be_overridden_from_the_environment(clean_env, monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "some/other-model")
    monkeypatch.setenv("GROUNDING_MODEL", "another/model")
    settings = Settings(_env_file=None)

    assert settings.generation_model == "some/other-model"
    assert settings.grounding_model == "another/model"
    assert settings.grader_model == "openai/gpt-oss-20b"


def test_models_can_be_set_in_an_env_file(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GRADER_MODEL=from/dotenv\n")

    assert Settings(_env_file=env_file).grader_model == "from/dotenv"
