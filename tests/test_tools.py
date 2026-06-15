"""Tests for LiteLLM-compatible financial tool functions."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fin_ai.core import tools


class _FakeTicker:
    def __init__(self, symbol: str) -> None:
        self.ticker = symbol
        self.info = {
            "shortName": "Acme Inc",
            "industry": "Technology",
            "sector": "Software",
            "country": "USA",
            "website": "https://acme.example",
            "marketCap": 123456,
        }
        self._history = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "Close": [102.0, 103.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )
        self.dividends = pd.Series(
            [0.25, 0.30], index=pd.to_datetime(["2026-01-01", "2026-04-01"])
        )
        self.financials = pd.DataFrame({"2025": [1000]}, index=["Revenue"])
        self.balance_sheet = pd.DataFrame({"2025": [500]}, index=["Assets"])
        self.cashflow = pd.DataFrame({"2025": [120]}, index=["Operating Cash Flow"])
        self.recommendations = pd.DataFrame(
            [{"period": "0m", "buy": 7, "hold": 2, "sell": 1}]
        )

    def history(self, start: str, end: str):
        return self._history


@pytest.fixture
def fake_ticker(monkeypatch):
    monkeypatch.setattr(tools.yf, "Ticker", _FakeTicker)


class TestLiteLlmToolFunctions:
    @pytest.mark.unit
    def test_get_stock_data_returns_json_serializable_payload(self, fake_ticker):
        result = tools.get_stock_data("AAPL", "2026-01-01", "2026-01-31")

        assert result["symbol"] == "AAPL"
        assert result["row_count"] == 2
        assert result["truncated"] is False
        assert result["records"][0]["index"] == "2026-01-02"
        assert result["records"][1]["Close"] == 103.0

    @pytest.mark.unit
    def test_dataframe_to_records_normalizes_timestamp_column_keys(self):
        frame = pd.DataFrame(
            {pd.Timestamp("2025-12-31"): [1000]},
            index=["Revenue"],
        )

        result = tools._dataframe_to_records(frame)

        assert result["records"][0]["index"] == "Revenue"
        assert result["records"][0]["2025-12-31T00:00:00"] == 1000
        json.dumps(result)

    @pytest.mark.unit
    def test_get_analyst_recommendations_returns_consensus(self, fake_ticker):
        result = tools.get_analyst_recommendations("MSFT")

        assert result["symbol"] == "MSFT"
        assert result["majority_recommendation"] == "buy"
        assert result["vote_count"] == 7
        assert result["has_recommendations"] is True


class TestLiteLlmToolRegistry:
    @pytest.mark.unit
    def test_tool_schemas_match_registered_functions(self):
        schema_names = {entry["function"]["name"] for entry in tools.YAHOO_FINANCE_TOOLS}
        registered_names = set(tools.LITELLM_TOOL_FUNCTIONS)

        assert schema_names == registered_names

    @pytest.mark.unit
    def test_execute_litellm_tool_call_dispatches_and_handles_errors(self, fake_ticker):
        ok = tools.execute_litellm_tool_call(
            "get_stock_info", {"symbol": "AAPL"}
        )
        unsupported = tools.execute_litellm_tool_call("no_such_tool", {})
        invalid_args = tools.execute_litellm_tool_call("get_stock_info", {})

        assert ok["shortName"] == "Acme Inc"
        assert unsupported["error"].startswith("Unsupported tool")
        assert invalid_args["error"].startswith("Invalid arguments")
