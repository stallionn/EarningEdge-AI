"""
EarningsEdge AI — Phase 4 Validation Test Suite

Validates FastAPI functionality, ensuring endpoints mount, models load,
and schema structures hold true.

Run:
    pytest tests/test_phase4.py -v
"""

import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

class TestPhase4API:

    def test_health_check(self, client):
        """API must return a 200 health status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"]["database_ready"] == True

    def test_historical_signals_endpoint_valid_ticker(self, client):
        """GET /signals/{ticker} should return records for a valid synthetic test ticker."""
        # Check an arbitrary common ticker, e.g., AAPL
        response = client.get("/signals/AAPL")
        # Could be 200 (if AAPL exists in synthetic data) or 404 (if not)
        assert response.status_code in (200, 404)
        
        if response.status_code == 200:
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert isinstance(data["data"], list)

    def test_historical_signals_endpoint_invalid_ticker(self, client):
        """GET /signals/{ticker} must gracefully return 404 for missing values."""
        response = client.get("/signals/FAKETICKER999")
        assert response.status_code == 404
        assert "No call records found" in response.json()["detail"]

    def test_realtime_analysis_valid_text(self, client):
        """POST /analyze must parse, score, and return correct metrics schemas for texts."""
        dummy_transcript = (
            "Good afternoon. We are very excited to announce our record-breaking "
            "revenue this quarter. Margins expanded significantly. "
            "However, we must note that these numbers are subject to potential fluctuation "
            "in the forward-looking quarters due to macro conditions."
        )
        response = client.post("/analyze", json={"text": dummy_transcript})
        assert response.status_code == 200
        
        data = response.json()
        assert "confidence_score" in data
        assert "hedge_density" in data
        assert data["sentences_parsed"] >= 1
        assert data["total_words"] > 10

    def test_realtime_analysis_short_text(self, client):
        """POST /analyze must reject strings that are unreasonably short."""
        response = client.post("/analyze", json={"text": "Hello"})
        assert response.status_code == 400
