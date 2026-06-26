"""Tests for the MarketDataService layer (online + offline)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fin_ai.core.exceptions import (
    MarketDataNotFoundError,
    MarketDataServiceError,
)
from fin_ai.core.service import (
    MarketDataService,
    OfflineMarketDataService,
    OnlineMarketDataService,
    _load_dataframe,
    _load_dict,
    _path_for,
)


# ===========================================================================
# Online service tests
# ===========================================================================


class FakeYfTicker:
    """Minimal yfinance Ticker stand-in for testing OnlineMarketDataService."""

    def __init__(self, symbol: str) -> None:
        self.ticker = symbol
        self.info = {
            "shortName": "TestCorp",
            "industry": "Robotics",
            "sector": "Technology",
            "country": "USA",
            "website": "https://testcorp.example",
        }
        self.dividends = pd.Series(
            [0.50, 0.55],
            index=pd.to_datetime(["2026-03-01", "2026-06-01"]),
            name="dividend",
        )
        self.financials = pd.DataFrame({"2025": [5000]}, index=["Total Revenue"])
        self.balance_sheet = pd.DataFrame({"2025": [10000]}, index=["Total Assets"])
        self.cashflow = pd.DataFrame({"2025": [800]}, index=["Free Cash Flow"])
        self.recommendations = pd.DataFrame(
            [{"period": "0m", "buy": 10, "hold": 1, "sell": 0}]
        )

    def history(self, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame(
            {"Open": [200.0], "Close": [205.0]},
            index=pd.to_datetime(["2026-06-01"]),
        )


class FakeYfTickerEmptyHistory:
    """Returns empty history DataFrame."""

    def history(self, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def info(self) -> dict:
        return {}

    @property
    def dividends(self) -> pd.Series:
        return pd.Series(dtype=float)

    @property
    def financials(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def balance_sheet(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def cashflow(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def recommendations(self) -> pd.DataFrame:
        return pd.DataFrame()


class FakeYfTickerFailing:
    """Simulates a network/API error."""

    def history(self, start: str, end: str) -> pd.DataFrame:
        raise ConnectionError("Connection refused")

    @property
    def info(self) -> dict:
        raise ConnectionError("Connection refused")


@pytest.fixture
def online_service(monkeypatch):
    svc = OnlineMarketDataService()
    monkeypatch.setattr(svc, "_get_ticker", lambda sym: FakeYfTicker(sym))
    return svc


@pytest.fixture
def online_service_empty(monkeypatch):
    svc = OnlineMarketDataService()
    monkeypatch.setattr(svc, "_get_ticker", lambda sym: FakeYfTickerEmptyHistory())
    return svc


@pytest.fixture
def online_service_failing(monkeypatch):
    svc = OnlineMarketDataService()
    monkeypatch.setattr(svc, "_get_ticker", lambda sym: FakeYfTickerFailing())
    return svc


class TestOnlineMarketDataService:
    @pytest.mark.unit
    def test_get_stock_data_returns_dataframe(self, online_service):
        df = online_service.get_stock_data("TEST", "2026-01-01", "2026-12-31")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "Close" in df.columns

    @pytest.mark.unit
    def test_get_stock_data_raises_on_empty(self, online_service_empty):
        with pytest.raises(MarketDataNotFoundError) as exc:
            online_service_empty.get_stock_data("TEST", "2026-01-01", "2026-12-31")
        assert "stock_data" in str(exc.value)

    @pytest.mark.unit
    def test_get_stock_data_raises_on_connection_error(self, online_service_failing):
        with pytest.raises(MarketDataServiceError) as exc:
            online_service_failing.get_stock_data("TEST", "2026-01-01", "2026-12-31")
        assert "stock_data" in str(exc.value)

    @pytest.mark.unit
    def test_get_stock_info_returns_dict(self, online_service):
        info = online_service.get_stock_info("TEST")
        assert isinstance(info, dict)
        assert info["shortName"] == "TestCorp"

    @pytest.mark.unit
    def test_get_stock_info_raises_on_empty(self, online_service_empty):
        with pytest.raises(MarketDataNotFoundError):
            online_service_empty.get_stock_info("TEST")

    @pytest.mark.unit
    def test_get_company_info_returns_dataframe(self, online_service):
        df = online_service.get_company_info("TEST")
        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["Company Name"] == "TestCorp"

    @pytest.mark.unit
    def test_get_stock_dividends_returns_dataframe(self, online_service):
        df = online_service.get_stock_dividends("TEST")
        assert isinstance(df, pd.DataFrame)
        assert "dividend" in df.columns or df.shape[1] >= 1

    @pytest.mark.unit
    def test_get_stock_dividends_raises_on_empty(self, online_service_empty):
        with pytest.raises(MarketDataNotFoundError):
            online_service_empty.get_stock_dividends("TEST")

    @pytest.mark.unit
    def test_get_income_stmt_returns_dataframe(self, online_service):
        df = online_service.get_income_stmt("TEST")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @pytest.mark.unit
    def test_get_balance_sheet_returns_dataframe(self, online_service):
        df = online_service.get_balance_sheet("TEST")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @pytest.mark.unit
    def test_get_cash_flow_returns_dataframe(self, online_service):
        df = online_service.get_cash_flow("TEST")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @pytest.mark.unit
    def test_get_analyst_recommendations_returns_dataframe(self, online_service):
        df = online_service.get_analyst_recommendations("TEST")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @pytest.mark.unit
    def test_get_analyst_recommendations_raises_on_empty(self, online_service_empty):
        with pytest.raises(MarketDataNotFoundError):
            online_service_empty.get_analyst_recommendations("TEST")


# ===========================================================================
# Offline service tests — real downloaded data
# ===========================================================================


@pytest.fixture(scope="module")
def real_data_dir() -> Path:
    """Path to the project's data directory containing pre-downloaded tickers."""
    return Path(__file__).resolve().parent.parent.parent / "data"


class TestOfflineMarketDataServiceReal:
    """Integration tests using real downloaded AAPL/MSFT data."""

    @pytest.fixture
    def offline_service(self, real_data_dir) -> OfflineMarketDataService:
        return OfflineMarketDataService(real_data_dir)

    @pytest.mark.integration
    def test_get_stock_data_returns_aapl(self, offline_service):
        df = offline_service.get_stock_data("AAPL", "2024-01-01", "2024-12-31")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "Close" in df.columns or "Open" in df.columns

    @pytest.mark.integration
    def test_get_stock_data_respects_date_range(self, offline_service):
        full = offline_service.get_stock_data("AAPL", "2024-01-01", "2024-12-31")
        narrowed = offline_service.get_stock_data("AAPL", "2024-06-01", "2024-06-30")
        assert len(narrowed) < len(full)

    @pytest.mark.integration
    def test_get_stock_info_returns_dict(self, offline_service):
        info = offline_service.get_stock_info("AAPL")
        assert isinstance(info, dict)
        # stock_info is stored as JSON via yfinance.info — should have some keys
        assert len(info) > 0

    @pytest.mark.integration
    def test_get_company_info_returns_dataframe(self, offline_service):
        df = offline_service.get_company_info("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @pytest.mark.integration
    def test_get_stock_dividends_returns_dataframe(self, offline_service):
        df = offline_service.get_stock_dividends("AAPL")
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.integration
    def test_get_income_stmt_returns_dataframe(self, offline_service):
        df = offline_service.get_income_stmt("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @pytest.mark.integration
    def test_get_balance_sheet_returns_dataframe(self, offline_service):
        df = offline_service.get_balance_sheet("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @pytest.mark.integration
    def test_get_cash_flow_returns_dataframe(self, offline_service):
        df = offline_service.get_cash_flow("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @pytest.mark.integration
    def test_get_analyst_recommendations_returns_dataframe(self, offline_service):
        df = offline_service.get_analyst_recommendations("AAPL")
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.integration
    def test_multiple_tickers(self, offline_service):
        aapl = offline_service.get_stock_info("AAPL")
        msft = offline_service.get_stock_info("MSFT")
        assert isinstance(aapl, dict)
        assert isinstance(msft, dict)
        assert aapl != msft

    @pytest.mark.integration
    def test_unknown_ticker_raises(self, offline_service):
        with pytest.raises(MarketDataNotFoundError):
            offline_service.get_stock_data("ZZZZ", "2024-01-01", "2024-12-31")


# ===========================================================================
# Offline service tests — synthetic temp directory
# ===========================================================================


class TestOfflineMarketDataServiceTemp:
    """Offline service tests with synthetic data written to a temp dir."""

    @pytest.fixture
    def temp_data_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def offline_service(self, temp_data_dir) -> OfflineMarketDataService:
        return OfflineMarketDataService(temp_data_dir)

    def _write_parquet(self, directory: Path, filename: str, df: pd.DataFrame):
        directory.mkdir(parents=True, exist_ok=True)
        df.to_parquet(directory / filename, index=True)

    @pytest.mark.unit
    def test_raises_when_ticker_dir_missing(self, offline_service, temp_data_dir):
        with pytest.raises(MarketDataNotFoundError):
            offline_service.get_stock_data("MISSING", "2024-01-01", "2024-12-31")

    @pytest.mark.unit
    def test_raises_when_file_missing(self, offline_service, temp_data_dir):
        (temp_data_dir / "TEST").mkdir(parents=True, exist_ok=True)
        with pytest.raises(MarketDataNotFoundError):
            offline_service.get_stock_data("TEST", "2024-01-01", "2024-12-31")

    @pytest.mark.unit
    def test_loads_correct_type(self, offline_service, temp_data_dir):
        df = pd.DataFrame(
            {"Close": [150.0]},
            index=pd.to_datetime(["2024-06-01"]),
        )
        self._write_parquet(temp_data_dir / "TEST", "stock_data.parquet", df)

        result = offline_service.get_stock_data("TEST", "2024-01-01", "2024-12-31")
        assert not result.empty
        assert result["Close"].iloc[0] == 150.0

    @pytest.mark.unit
    def test_stock_info_from_json(self, offline_service, temp_data_dir):
        import json

        ticker_dir = temp_data_dir / "TEST"
        ticker_dir.mkdir(parents=True, exist_ok=True)
        (ticker_dir / "stock_info.json").write_text(
            json.dumps({"shortName": "JSON Corp"}), encoding="utf-8"
        )

        info = offline_service.get_stock_info("TEST")
        assert info["shortName"] == "JSON Corp"

    @pytest.mark.unit
    def test_csv_dataframe(self, offline_service, temp_data_dir):
        ticker_dir = temp_data_dir / "CSVTEST"
        ticker_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2024-01-01"])).to_csv(
            ticker_dir / "stock_data.csv", index=True
        )

        result = offline_service.get_stock_data("CSVTEST", "2024-01-01", "2024-12-31")
        assert not result.empty

    @pytest.mark.unit
    def test_pkl_dataframe(self, offline_service, temp_data_dir):
        ticker_dir = temp_data_dir / "PKLTEST"
        ticker_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2024-01-01"]))
        pd.to_pickle(df, ticker_dir / "stock_data.pkl")

        result = offline_service.get_stock_data("PKLTEST", "2024-01-01", "2024-12-31")
        assert not result.empty


# ===========================================================================
# Factory tests
# ===========================================================================


class TestMarketDataServiceFactory:
    @pytest.mark.unit
    def test_from_environment_returns_offline_when_data_exists(self, monkeypatch):
        monkeypatch.setattr(
            "fin_ai.core.service.YAHOO_SERVICE_OFFLINE", True
        )
        with tempfile.TemporaryDirectory() as tmp:
            # Create a non-empty data dir
            (Path(tmp) / "DUMMY").mkdir()
            monkeypatch.setattr(
                "fin_ai.core.service.YAHOO_DATA_DIR", tmp
            )
            svc = MarketDataService.from_environment()
            assert isinstance(svc, OfflineMarketDataService)

    @pytest.mark.unit
    def test_from_environment_returns_online_when_offline_false(self, monkeypatch):
        monkeypatch.setattr(
            "fin_ai.core.service.YAHOO_SERVICE_OFFLINE", False
        )
        svc = MarketDataService.from_environment()
        assert isinstance(svc, OnlineMarketDataService)

    @pytest.mark.unit
    def test_from_environment_falls_back_to_online_when_data_empty(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "fin_ai.core.service.YAHOO_SERVICE_OFFLINE", True
        )
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(
                "fin_ai.core.service.YAHOO_DATA_DIR", tmp
            )
            svc = MarketDataService.from_environment()
            assert isinstance(svc, OnlineMarketDataService)


# ===========================================================================
# Internal helper tests
# ===========================================================================


class TestInternalHelpers:
    @pytest.mark.unit
    def test_path_for_returns_none_when_dir_missing(self, tmp_path):
        result = _path_for(tmp_path, "NONEXIST", "stock_data")
        assert result is None

    @pytest.mark.unit
    def test_path_for_finds_parquet(self, tmp_path):
        ticker_dir = tmp_path / "TEST"
        ticker_dir.mkdir()
        (ticker_dir / "stock_data.parquet").touch()
        result = _path_for(tmp_path, "TEST", "stock_data")
        assert result is not None
        assert result.suffix == ".parquet"

    @pytest.mark.unit
    def test_path_for_prefers_first_ext_in_order(self, tmp_path):
        ticker_dir = tmp_path / "PREF"
        ticker_dir.mkdir()
        (ticker_dir / "stock_data.parquet").touch()
        (ticker_dir / "stock_data.csv").touch()
        result = _path_for(tmp_path, "PREF", "stock_data")
        # .parquet is first in _SERIALISATION_EXTENSIONS
        assert result is not None
        assert result.suffix == ".parquet"

    @pytest.mark.unit
    def test_load_dataframe_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=True)
        loaded = _load_dataframe(path)
        assert list(loaded["a"]) == [1, 2]

    @pytest.mark.unit
    def test_load_dataframe_json_list(self, tmp_path):
        import json

        path = tmp_path / "test.json"
        path.write_text(json.dumps([{"x": 10}, {"x": 20}]), encoding="utf-8")
        loaded = _load_dataframe(path)
        assert list(loaded["x"]) == [10, 20]

    @pytest.mark.unit
    def test_load_dataframe_json_dict(self, tmp_path):
        import json

        path = tmp_path / "test.json"
        path.write_text(json.dumps({"name": "Single"}), encoding="utf-8")
        loaded = _load_dataframe(path)
        assert loaded.iloc[0]["name"] == "Single"

    @pytest.mark.unit
    def test_load_dict(self, tmp_path):
        import json

        path = tmp_path / "info.json"
        path.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        data = _load_dict(path)
        assert data["key"] == "value"

    @pytest.mark.unit
    def test_load_dataframe_unknown_ext_raises(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.touch()
        with pytest.raises(MarketDataServiceError):
            _load_dataframe(path)
