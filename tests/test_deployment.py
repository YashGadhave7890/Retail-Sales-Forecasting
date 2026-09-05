"""
Unit tests for deployment configuration, Docker setup, and security boundaries.

Validates:
1. Dockerfile exists, uses Python 3.11-slim, and properly configures Streamlit.
2. .dockerignore exists and excludes sensitive directories (.venv, .git, .env, scratch).
3. Required dashboard application files exist.
4. Production model artifact and metadata exist.
5. requirements.txt contains all core analytical and web dependencies.
6. Streamlit configuration file (.streamlit/config.toml) is valid.
7. No secret or credential files are present in the repository.
"""
from pathlib import Path
import pytest

from src.config import PROJECT_ROOT, MODELS_DIR

DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
STREAMLIT_CONFIG_PATH = PROJECT_ROOT / ".streamlit" / "config.toml"
DASHBOARD_APP_PATH = PROJECT_ROOT / "dashboard" / "app.py"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


def test_dockerfile_exists_and_configured():
    """Verify Dockerfile exists and contains required production directives."""
    assert DOCKERFILE_PATH.exists(), "Dockerfile missing!"
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    
    assert "FROM python:3.11-slim" in content, "Dockerfile should use python:3.11-slim"
    assert "EXPOSE 8501" in content, "Dockerfile must expose port 8501"
    assert "WORKDIR /app" in content, "Dockerfile must set working directory"
    assert "HEALTHCHECK" in content, "Dockerfile must define container health check"
    assert "dashboard/app.py" in content, "Dockerfile entrypoint must point to dashboard/app.py"
    assert "--server.headless=true" in content, "Dockerfile must configure headless Streamlit"
    assert "--server.address=0.0.0.0" in content, "Dockerfile must bind to 0.0.0.0"


def test_dockerignore_exists_and_excludes_sensitive():
    """Verify .dockerignore excludes virtual environments, git, caches, and secrets."""
    assert DOCKERIGNORE_PATH.exists(), ".dockerignore missing!"
    content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    
    required_exclusions = [".venv", "__pycache__", ".git", ".env", "scratch"]
    for exclusion in required_exclusions:
        assert exclusion in content, f".dockerignore must exclude '{exclusion}'"


def test_required_dashboard_files_exist():
    """Verify dashboard/app.py and helper modules exist."""
    assert DASHBOARD_APP_PATH.exists(), "dashboard/app.py missing!"
    assert (PROJECT_ROOT / "dashboard" / "data_utils.py").exists(), "dashboard/data_utils.py missing!"
    assert (PROJECT_ROOT / "dashboard" / "README.md").exists(), "dashboard/README.md missing!"


def test_production_model_exists():
    """Verify production model artifact and metadata exist and are populated."""
    assert FINAL_MODEL_PATH.exists(), "models/final_model.joblib missing!"
    assert FINAL_MODEL_PATH.stat().st_size > 1000, "final_model.joblib is empty!"
    assert MODEL_METADATA_PATH.exists(), "models/model_metadata.json missing!"


def test_requirements_txt_exists_and_complete():
    """Verify requirements.txt contains all necessary packages."""
    assert REQUIREMENTS_PATH.exists(), "requirements.txt missing!"
    reqs = REQUIREMENTS_PATH.read_text(encoding="utf-8").lower()
    
    expected_pkgs = ["streamlit", "scikit-learn", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "altair"]
    for pkg in expected_pkgs:
        assert pkg in reqs, f"requirements.txt missing expected package: '{pkg}'"


def test_streamlit_config_toml_valid():
    """Verify .streamlit/config.toml exists and configures server correctly."""
    assert STREAMLIT_CONFIG_PATH.exists(), ".streamlit/config.toml missing!"
    content = STREAMLIT_CONFIG_PATH.read_text(encoding="utf-8")
    assert "headless = true" in content
    assert "port = 8501" in content
    assert 'address = "0.0.0.0"' in content


def test_no_secret_files_committed():
    """Verify no active .env or secret files exist in the repository."""
    env_file = PROJECT_ROOT / ".env"
    assert not env_file.exists(), "Active .env file detected! Secrets must not be committed."
    
    gitignore_file = PROJECT_ROOT / ".gitignore"
    assert gitignore_file.exists(), ".gitignore missing!"
    gi_content = gitignore_file.read_text(encoding="utf-8")
    assert ".env" in gi_content, ".gitignore must ignore .env files"
