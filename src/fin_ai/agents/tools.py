
import yfinance as yf
from typing import Any, Optional
import pandas as pd
from pandas import DataFrame


def _get_ticker(symbol: str) -> yf.Ticker:
    """Create a yfinance ticker instance from a ticker symbol."""
    return yf.Ticker(symbol)


def _to_json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _dataframe_to_records(frame: DataFrame, max_rows: int = 200) -> dict:
    if frame is None or frame.empty:
        return {"row_count": 0, "truncated": False, "records": []}

    converted = frame.copy()
    if isinstance(converted.index, pd.DatetimeIndex):
        converted.index = converted.index.strftime("%Y-%m-%d")

    records = converted.reset_index().to_dict(orient="records")
    json_records = [
        {k: _to_json_value(v) for k, v in row.items()}
        for row in records[:max_rows]
    ]
    return {
        "row_count": len(records),
        "truncated": len(records) > max_rows,
        "records": json_records,
    }


def get_stock_data(symbol: str, start_date: str, end_date: str) -> dict:
    """Retrieve stock price data for a ticker symbol within the date range."""
    ticker = _get_ticker(symbol)
    stock_data = ticker.history(start=start_date, end=end_date)
    payload = _dataframe_to_records(stock_data)
    payload.update({"symbol": symbol, "start_date": start_date, "end_date": end_date})
    return payload


def get_stock_info(symbol: str) -> dict:
    """Fetches and returns latest stock information."""
    ticker = _get_ticker(symbol)
    stock_info = ticker.info
    return {k: _to_json_value(v) for k, v in stock_info.items()}


def get_company_info(symbol: str, save_path: Optional[str] = None) -> dict:
    """Fetches and returns company information as a DataFrame."""
    ticker = _get_ticker(symbol)
    info = ticker.info
    company_info = {
        "Company Name": info.get("shortName", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Sector": info.get("sector", "N/A"),
        "Country": info.get("country", "N/A"),
        "Website": info.get("website", "N/A"),
    }
    company_info_df = DataFrame([company_info])
    if save_path:
        company_info_df.to_csv(save_path)
        print(f"Company info for {ticker.ticker} saved to {save_path}")
    return {
        "symbol": symbol,
        "company_info": company_info,
        "saved_to": save_path,
    }


def get_stock_dividends(symbol: str, save_path: Optional[str] = None) -> dict:
    """Fetches and returns the latest dividends data as a DataFrame."""
    ticker = _get_ticker(symbol)
    dividends = ticker.dividends
    if save_path:
        dividends.to_csv(save_path)
        print(f"Dividends for {ticker.ticker} saved to {save_path}")
    payload = _dataframe_to_records(dividends.to_frame(name="dividend"))
    payload.update({"symbol": symbol, "saved_to": save_path})
    return payload


def get_income_stmt(symbol: str) -> dict:
    """Fetches and returns the latest income statement of the company as a DataFrame."""
    ticker = _get_ticker(symbol)
    income_stmt = ticker.financials
    payload = _dataframe_to_records(income_stmt)
    payload.update({"symbol": symbol})
    return payload


def get_balance_sheet(symbol: str) -> dict:
    """Fetches and returns the latest balance sheet of the company as a DataFrame."""
    ticker = _get_ticker(symbol)
    balance_sheet = ticker.balance_sheet
    payload = _dataframe_to_records(balance_sheet)
    payload.update({"symbol": symbol})
    return payload


def get_cash_flow(symbol: str) -> dict:
    """Fetches and returns the latest cash flow statement of the company as a DataFrame."""
    ticker = _get_ticker(symbol)
    cash_flow = ticker.cashflow
    payload = _dataframe_to_records(cash_flow)
    payload.update({"symbol": symbol})
    return payload


def get_analyst_recommendations(symbol: str) -> dict:
    """Fetches the latest analyst recommendations and returns the most common recommendation and its count."""
    ticker = _get_ticker(symbol)
    recommendations = ticker.recommendations
    if recommendations.empty:
        return {
            "symbol": symbol,
            "majority_recommendation": None,
            "vote_count": 0,
            "has_recommendations": False,
        }

    row_0 = recommendations.iloc[0, 1:]
    max_votes = row_0.max()
    majority_voting_result = row_0[row_0 == max_votes].index.tolist()

    return {
        "symbol": symbol,
        "majority_recommendation": majority_voting_result[0],
        "vote_count": _to_json_value(max_votes),
        "has_recommendations": True,
    }


LITELLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Retrieve historical stock prices for a ticker symbol in a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                },
                "required": ["symbol", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_info",
            "description": "Get the latest stock metadata and quote information for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Get basic company profile fields for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                    "save_path": {"type": "string", "description": "Optional CSV output path"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_dividends",
            "description": "Get dividend history for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                    "save_path": {"type": "string", "description": "Optional CSV output path"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_stmt",
            "description": "Get the most recent income statement for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Get the most recent balance sheet for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cash_flow",
            "description": "Get the most recent cash-flow statement for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_analyst_recommendations",
            "description": "Get the latest analyst consensus recommendation summary for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., AAPL"},
                },
                "required": ["symbol"],
            },
        },
    },
]


LITELLM_TOOL_FUNCTIONS = {
    "get_stock_data": get_stock_data,
    "get_stock_info": get_stock_info,
    "get_company_info": get_company_info,
    "get_stock_dividends": get_stock_dividends,
    "get_income_stmt": get_income_stmt,
    "get_balance_sheet": get_balance_sheet,
    "get_cash_flow": get_cash_flow,
    "get_analyst_recommendations": get_analyst_recommendations,
}


def execute_litellm_tool_call(name: str, arguments: dict[str, Any]) -> dict:
    """Execute a LiteLLM tool by name using decoded tool-call arguments."""
    func = LITELLM_TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unsupported tool: {name}"}

    try:
        return func(**arguments)
    except TypeError as exc:
        return {"error": f"Invalid arguments for {name}: {exc}"}
    except Exception as exc:
        return {"error": str(exc)}

