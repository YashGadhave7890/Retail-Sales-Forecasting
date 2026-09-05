"""
Unit tests for src.validate_data module.
"""
from pathlib import Path
import pytest

from src.validate_data import validate_raw_data, generate_markdown_report, compute_file_hash
from src.config import RAW_DATA_FILE

EXPECTED_SHA256: str = "c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce"


def test_compute_file_hash():
    """Verify cryptographic hash calculation matches the recorded raw CSV hash."""
    file_hash = compute_file_hash(RAW_DATA_FILE)
    assert file_hash == EXPECTED_SHA256, (
        f"Raw file hash changed! Expected {EXPECTED_SHA256}, got {file_hash}"
    )


def test_validate_raw_data_structure():
    """Verify validate_raw_data returns expected dictionary keys and structures."""
    results = validate_raw_data()
    assert isinstance(results, dict)
    expected_sections = [
        "file_info", "dimensions", "schema", "missing_values",
        "duplicates", "dates", "numerics", "categoricals"
    ]
    for section in expected_sections:
        assert section in results, f"Missing section '{section}' in validation results"


def test_validation_integrity_passes():
    """Verify that fundamental integrity checks all pass on the raw dataset."""
    results = validate_raw_data()
    assert results["dimensions"]["passed"] is True
    assert results["schema"]["passed"] is True
    assert results["missing_values"]["passed"] is True
    assert results["duplicates"]["passed"] is True
    assert results["dates"]["passed"] is True
    assert results["numerics"]["passed"] is True
    assert results["categoricals"]["passed"] is True


def test_date_chronology_invariant():
    """Verify that no order occurs after its ship date."""
    results = validate_raw_data()
    assert results["dates"]["order_after_ship_count"] == 0
    assert results["dates"]["min_shipping_days"] >= 0


def test_missing_file_validation_raises_error(tmp_path: Path):
    """Verify that validate_raw_data raises FileNotFoundError for missing path."""
    non_existent = tmp_path / "missing_data.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset file not found at"):
        validate_raw_data(file_path=non_existent)


def test_generate_markdown_report(tmp_path: Path):
    """Verify report generator produces valid markdown content and writes to file."""
    results = validate_raw_data()
    output_file = tmp_path / "test_validation_report.md"
    content = generate_markdown_report(results, output_path=output_file)
    assert isinstance(content, str)
    assert len(content) > 500
    assert "# Raw Data Validation Report" in content
    assert output_file.exists()
    assert output_file.stat().st_size > 500
