#!/usr/bin/env python3
"""
Offline data downloader for Yahoo Finance financial tools.

Downloads all data types used by ``fin_ai.core.tools`` functions into a local
directory so that the tools can be used **offline** by serving the saved files.

Each ticker gets its own subfolder under a top-level data directory::

    <data_dir>/
    └── <SYMBOL>/
        ├── stock_data.<ext>
        ├── stock_info.<ext>
        ├── company_info.<ext>
        ├── dividends.<ext>
        ├── income_stmt.<ext>
        ├── balance_sheet.<ext>
        ├── cash_flow.<ext>
        └── analyst_recommendations.<ext>

Usage examples
--------------
Download everything for AAPL and MSFT to ``data/`` as CSV files::

    python scripts/yahoo_finance_download.py --tickers AAPL,MSFT \\
        --start 2024-01-01 --end 2025-12-31 --format csv

Download only price data for NVDA as Parquet::

    python scripts/yahoo_finance_download.py --tickers NVDA \\
        --start 2024-06-01 --end 2025-06-01 \\
        --types stock_data --format parquet

Download all data types as JSON::

    python scripts/yahoo_finance_download.py --tickers AAPL,GOOGL,MSFT \\
        --start 2024-01-01 --end 2025-12-31 --format json --data-dir ./data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Supported format extensions
# ---------------------------------------------------------------------------

SERIALISERS: dict[str, tuple[str, str]] = {
    "json": ("json", "json"),
    "csv": ("csv", "csv"),
    "pkl": ("pkl", "pkl"),
    "parquet": ("parquet", "parquet"),
}

# ---------------------------------------------------------------------------
# Data-type catalog — mirrors the functions in fin_ai/core/tools.py
# ---------------------------------------------------------------------------

DATA_TYPES = {
    "stock_data": {
        "help": "Historical OHLCV price data (ticker.history)",
        "requires_range": True,
    },
    "stock_info": {
        "help": "Stock metadata / quote info (ticker.info)",
        "requires_range": False,
    },
    "company_info": {
        "help": "Company profile (name, industry, sector, …)",
        "requires_range": False,
    },
    "dividends": {
        "help": "Dividend history (ticker.dividends)",
        "requires_range": False,
    },
    "income_stmt": {
        "help": "Income statement (ticker.financials)",
        "requires_range": False,
    },
    "balance_sheet": {
        "help": "Balance sheet (ticker.balance_sheet)",
        "requires_range": False,
    },
    "cash_flow": {
        "help": "Cash flow statement (ticker.cashflow)",
        "requires_range": False,
    },
    "analyst_recommendations": {
        "help": "Analyst consensus recommendations (ticker.recommendations)",
        "requires_range": False,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated list of ticker symbols, e.g. AAPL,MSFT,NVDA",
    )
    parser.add_argument(
        "--start",
        default="2000-01-01",
        help="Start date in YYYY-MM-DD format (default: 2000-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Root directory for downloaded data. "
            "Defaults to <project_root>/data"
        ),
    )
    parser.add_argument(
        "--format",
        default="parquet",
        choices=list(SERIALISERS),
        help="Output format for all saved files (default: parquet)",
    )
    parser.add_argument(
        "--types",
        default=None,
        help=(
            "Comma-separated list of data types to download. "
            f"Choices: {', '.join(DATA_TYPES)}. "
            "Default: all types."
        ),
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data fetchers — one per type, returning a pandas object
# ---------------------------------------------------------------------------


def _fetch_stock_data(
    ticker: yf.Ticker,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    return ticker.history(start=start, end=end)


def _fetch_stock_info(ticker: yf.Ticker, **_: Any) -> dict:
    # info is a dict — wrap it so it can be serialised uniformly
    return ticker.info


def _fetch_company_info(ticker: yf.Ticker, **_: Any) -> pd.DataFrame:
    info = ticker.info
    return pd.DataFrame([
        {
            "Company Name": info.get("shortName", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Country": info.get("country", "N/A"),
            "Website": info.get("website", "N/A"),
        }
    ])


def _fetch_dividends(ticker: yf.Ticker, **_: Any) -> pd.DataFrame:
    div = ticker.dividends
    if isinstance(div, pd.Series):
        return div.to_frame(name="dividend")
    return div


def _fetch_income_stmt(ticker: yf.Ticker, **_: Any) -> pd.DataFrame:
    return ticker.financials


def _fetch_balance_sheet(ticker: yf.Ticker, **_: Any) -> pd.DataFrame:
    return ticker.balance_sheet


def _fetch_cash_flow(ticker: yf.Ticker, **_: Any) -> pd.DataFrame:
    return ticker.cashflow


def _fetch_analyst_recommendations(ticker: yf.Ticker, **_: Any) -> pd.DataFrame:
    return ticker.recommendations


_FETCHERS: dict[str, Any] = {
    "stock_data": _fetch_stock_data,
    "stock_info": _fetch_stock_info,
    "company_info": _fetch_company_info,
    "dividends": _fetch_dividends,
    "income_stmt": _fetch_income_stmt,
    "balance_sheet": _fetch_balance_sheet,
    "cash_flow": _fetch_cash_flow,
    "analyst_recommendations": _fetch_analyst_recommendations,
}

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _save_object(
    obj: Any,
    filepath: Path,
    fmt: str,
) -> None:
    """Save a DataFrame or dict to *filepath* in the requested *fmt*."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        if isinstance(obj, pd.DataFrame):
            data = _dataframe_to_json_safe(obj)
        elif isinstance(obj, dict):
            data = obj
        else:
            data = str(obj)
        filepath.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    elif fmt == "csv":
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(filepath, index=True)
        elif isinstance(obj, dict):
            pd.DataFrame([obj]).to_csv(filepath, index=False)
        else:
            filepath.write_text(str(obj), encoding="utf-8")
    elif fmt == "pkl":
        pd.to_pickle(obj, filepath)
    elif fmt == "parquet":
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(filepath, index=True)
        elif isinstance(obj, dict):
            pd.DataFrame([obj]).to_parquet(filepath, index=False)
        else:
            pd.DataFrame({"value": [obj]}).to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _dataframe_to_json_safe(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a JSON-safe list-of-dicts."""
    converted = df.copy()
    if isinstance(converted.index, pd.DatetimeIndex):
        converted.index = converted.index.strftime("%Y-%m-%d")
    records = converted.reset_index().to_dict(orient="records")
    # Convert any remaining non-serialisable values
    return json.loads(json.dumps(records, default=str, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------


def download_ticker(
    ticker_symbol: str,
    start: str,
    end: str | None,
    types: list[str],
    data_dir: Path,
    fmt: str,
) -> dict[str, str | None]:
    """Download selected data types for a single ticker.

    Returns a dict mapping type name → saved filepath (or ``None`` if failed).
    """
    ext = SERIALISERS[fmt][0]
    ticker = yf.Ticker(ticker_symbol)
    ticker_dir = data_dir / ticker_symbol.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str | None] = {}

    for dtype in types:
        fetcher = _FETCHERS[dtype]
        save_path = ticker_dir / f"{dtype}.{ext}"
        requires_range = DATA_TYPES[dtype]["requires_range"]

        try:
            if requires_range:
                obj = fetcher(ticker, start=start, end=end)
            else:
                obj = fetcher(ticker, start=start, end=end)

            # Some fetchers may return empty DataFrames
            if isinstance(obj, (pd.DataFrame, pd.Series)) and obj.empty:
                print(
                    f"  [WARN] {ticker_symbol} → {dtype}: empty result, "
                    f"skipping"
                )
                results[dtype] = None
                continue

            _save_object(obj, save_path, fmt)
            print(f"  [OK]   {ticker_symbol} → {save_path}")
            results[dtype] = str(save_path)

        except Exception as exc:
            print(f"  [FAIL] {ticker_symbol} → {dtype}: {exc}")
            results[dtype] = None

    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        # Default: project-root / data
        data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("ERROR: no tickers provided.", file=sys.stderr)
        return 1

    if args.types:
        selected_types = [t.strip() for t in args.types.split(",")]
        unknown = set(selected_types) - set(DATA_TYPES)
        if unknown:
            print(
                f"ERROR: unknown data type(s): {', '.join(sorted(unknown))}. "
                f"Choices: {', '.join(DATA_TYPES)}",
                file=sys.stderr,
            )
            return 1
    else:
        selected_types = list(DATA_TYPES)

    fmt = args.format
    print(f"Data directory : {data_dir}")
    print(f"Format         : {fmt}")
    print(f"Date range     : {args.start} → {args.end or 'today'}")
    print(f"Tickers        : {', '.join(tickers)}")
    print(f"Data types     : {', '.join(selected_types)}")
    print("-" * 60)

    summary: dict[str, dict[str, str | None]] = {}
    errors = 0

    for symbol in tickers:
        print(f"\n── {symbol} ──")
        result = download_ticker(
            ticker_symbol=symbol,
            start=args.start,
            end=args.end,
            types=selected_types,
            data_dir=data_dir,
            fmt=fmt,
        )
        summary[symbol] = result
        errors += sum(1 for v in result.values() if v is None)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for symbol, files in summary.items():
        ok = sum(1 for v in files.values() if v is not None)
        fail = sum(1 for v in files.values() if v is None)
        status = "✓" if fail == 0 else f"✗ ({fail} failed)"
        print(f"  {symbol}: {ok}/{len(files)} files  {status}")

    if errors:
        print(f"\n{errors} file(s) failed to download.")
        return 1

    print("\nAll downloads completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
