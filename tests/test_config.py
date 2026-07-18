from src.config import Settings, settings


def test_settings_loaded():
    assert settings.grid_eso_api_base
    assert settings.met_office_api_base
    assert settings.llm_provider in ("deepseek", "gemini")
    assert settings.seed == 42


def test_validate_required_flags_missing_keys():
    empty = Settings(_env_file=None, met_office_api_key="", wandb_api_key="", llm_api_key="")
    missing = empty.validate_required()
    assert set(missing) == {"MET_OFFICE_API_KEY", "WANDB_API_KEY", "LLM_API_KEY"}


def test_validate_required_passes_when_set():
    full = Settings(_env_file=None, met_office_api_key="k", wandb_api_key="k", llm_api_key="k")
    assert full.validate_required() == []
