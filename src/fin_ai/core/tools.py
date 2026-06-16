
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


def _to_json_key(value: Any) -> str | int | float | bool | None:
    normalized = _to_json_value(value)
    if isinstance(normalized, (str, int, float, bool)) or normalized is None:
        return normalized
    return str(normalized)


def _dataframe_to_records(frame: DataFrame, max_rows: int = 200) -> dict:
    if frame is None or frame.empty:
        return {"row_count": 0, "truncated": False, "records": []}

    converted = frame.copy()
    if isinstance(converted.index, pd.DatetimeIndex):
        converted.index = converted.index.strftime("%Y-%m-%d")

    records = converted.reset_index().to_dict(orient="records")
    json_records = [
        {_to_json_key(k): _to_json_value(v) for k, v in row.items()}
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


# ---------------------------------------------------------------------------
# Research publishing — PDF / HTML generation + email distribution
# ---------------------------------------------------------------------------

import json as _json
import os as _os
import smtplib as _smtplib
from datetime import datetime as _datetime
from email.mime.text import MIMEText as _MIMEText
from email.mime.multipart import MIMEMultipart as _MIMEMultipart
from email.mime.base import MIMEBase as _MIMEBase
from email import encoders as _encoders
from pathlib import Path as _Path

_OUTPUT_DIR = _Path(_os.getcwd()) / "published_research"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 20px;
         color: #1a1a2e; line-height: 1.7; background: #fafafa; }
  h1 { color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }
  h2 { color: #0f3460; margin-top: 30px; }
  h3 { color: #533483; }
  .meta { color: #666; font-size: 0.9em; margin-bottom: 30px; }
  .disclaimer { border-top: 1px solid #ccc; margin-top: 40px; padding-top: 15px;
                font-size: 0.8em; color: #888; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
  th { background: #0f3460; color: white; }
  tr:nth-child(even) { background: #f2f2f2; }
  pre { background: #1a1a2e; color: #e0e0e0; padding: 15px; border-radius: 8px;
        overflow-x: auto; }
  code { background: #eee; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
  blockquote { border-left: 4px solid #0f3460; margin: 15px 0; padding: 10px 20px;
               background: #f0f0f5; }
  @media print { body { max-width: 100%; } }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div class="meta">
  Generated: {{ date }} | FinAI Research Publisher
</div>
{{ content }}
<div class="disclaimer">
  This research is generated by an AI-powered agent.  It does not constitute
  financial advice.  Verify all data points before making investment decisions.
</div>
</body>
</html>"""


def _md_to_html(content: str) -> str:
    """Convert Markdown content to HTML, falling back to <pre> if no markers."""
    if any(marker in content for marker in ("#", "##", "**", "```", "- ", "* ")):
        try:
            import markdown as _mdlib
            return _mdlib.markdown(
                content,
                extensions=["tables", "fenced_code", "codehilite", "nl2br"],
            )
        except ImportError:
            # Fallback: use markdown-it-py (already in requirements.txt)
            from markdown_it import MarkdownIt
            md = MarkdownIt("commonmark", {"breaks": True, "html": True})
            return md.render(content)
    return f"<pre>{content}</pre>"


def _render_html(title: str, html_body: str) -> str:
    """Render full HTML page from title and body HTML."""
    from jinja2 import Template as _Template
    date_str = _datetime.now().strftime("%Y-%m-%d %H:%M")
    template = _Template(_HTML_TEMPLATE)
    return template.render(title=title, date=date_str, content=html_body)


def _safe_filename(title: str) -> str:
    """Sanitize title into a safe filename prefix."""
    safe = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()
    return safe[:80] if safe else "research_report"


def publish_research_html(content: str, title: str = "Research Report") -> str:
    """Generate a professional HTML research report and save it locally.

    Parameters
    ----------
    content : str
        Full research content (Markdown or plain text).
    title : str
        Report title displayed in the header.
    """
    html_body = _md_to_html(content)
    safe_title = _safe_filename(title)
    html = _render_html(title, html_body)

    filename = f"{safe_title.replace(' ', '_')}_{_datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    filepath = _OUTPUT_DIR / filename
    filepath.write_text(html, encoding="utf-8")

    return _json.dumps({
        "status": "published",
        "format": "html",
        "filepath": str(filepath),
        "filename": filename,
        "title": title,
    }, indent=2)


def publish_research_pdf(content: str, title: str = "Research Report") -> str:
    """Generate a PDF research report (rendered from HTML via weasyprint).

    Falls back to a print-ready HTML file if weasyprint is not installed.

    Parameters
    ----------
    content : str
        Full research content (Markdown supported).
    title : str
        Report title.
    """
    html_body = _md_to_html(content)
    safe_title = _safe_filename(title)
    html = _render_html(title, html_body)

    timestamp = _datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = safe_title.replace(" ", "_")

    try:
        from weasyprint import HTML as _WHTML
        pdf_path = _OUTPUT_DIR / f"{prefix}_{timestamp}.pdf"
        _WHTML(string=html).write_pdf(str(pdf_path))
        return _json.dumps({
            "status": "published",
            "format": "pdf",
            "filepath": str(pdf_path),
            "filename": pdf_path.name,
            "title": title,
            "engine": "weasyprint",
        }, indent=2)
    except ImportError:
        pass

    html_path = _OUTPUT_DIR / f"{prefix}_{timestamp}_printable.html"
    html_path.write_text(html, encoding="utf-8")

    return _json.dumps({
        "status": "published",
        "format": "html (print-to-PDF ready)",
        "filepath": str(html_path),
        "filename": html_path.name,
        "title": title,
        "note": "Open in browser and Ctrl+P / Cmd+P to save as PDF.",
    }, indent=2)


def send_research_email(
    recipient: str,
    subject: str,
    body: str,
    attachment_path: str = "",
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_password: str = "",
) -> str:
    """Send research report via email with optional file attachment.

    SMTP credentials are read from environment variables by default:
    ``AI_RESEARCH_SMTP_HOST``, ``AI_RESEARCH_SMTP_PORT``, ``AI_RESEARCH_SMTP_USER``,
    ``AI_RESEARCH_SMTP_PASSWORD``.  Override by passing arguments directly.

    Parameters
    ----------
    recipient : str
        Email address of the recipient.
    subject : str
        Email subject line.
    body : str
        Email body (Markdown — converted to HTML automatically).
    attachment_path : str
        Optional path to a file to attach.
    smtp_host : str
        SMTP server hostname.  Default: env ``AI_RESEARCH_SMTP_HOST``.
    smtp_port : int
        SMTP port.  Default: env ``AI_RESEARCH_SMTP_PORT`` or 587.
    smtp_user : str
        SMTP username.  Default: env ``AI_RESEARCH_SMTP_USER``.
    smtp_password : str
        SMTP password.  Default: env ``AI_RESEARCH_SMTP_PASSWORD``.
    """
    host = smtp_host or _os.getenv("AI_RESEARCH_SMTP_HOST", "")
    port = smtp_port if smtp_port != 587 else int(_os.getenv("AI_RESEARCH_SMTP_PORT", "587"))
    user = smtp_user or _os.getenv("AI_RESEARCH_SMTP_USER", "")
    password = smtp_password or _os.getenv("AI_RESEARCH_SMTP_PASSWORD", "")

    if not host:
        return _json.dumps({
            "status": "not_sent",
            "error": (
                "SMTP not configured.  Set AI_RESEARCH_SMTP_HOST, AI_RESEARCH_SMTP_USER, "
                "and AI_RESEARCH_SMTP_PASSWORD environment variables."
            ),
        }, indent=2)

    html_body = _md_to_html(body)
    msg = _MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user or "research@fin-ai.local"
    msg["To"] = recipient
    msg.attach(_MIMEText(html_body, "html", "utf-8"))

    if attachment_path and _Path(attachment_path).is_file():
        with open(attachment_path, "rb") as fh:
            part = _MIMEBase("application", "octet-stream")
            part.set_payload(fh.read())
            _encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{_Path(attachment_path).name}"',
            )
            msg.attach(part)

    try:
        with _smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return _json.dumps({
            "status": "sent",
            "recipient": recipient,
            "subject": subject,
            "attachment": attachment_path if attachment_path else None,
        }, indent=2)
    except Exception as exc:
        return _json.dumps({"status": "failed", "error": str(exc)}, indent=2)


def publish_research_report(
    content: str,
    title: str = "Research Report",
    format: str = "html",
    email: str = "",
) -> str:
    """Publish a research report — generate HTML/PDF and optionally email it.

    This is the primary publishing tool.  Chains: format → save → email.

    Parameters
    ----------
    content : str
        Full research content (Markdown format recommended).
    title : str
        Report title.
    format : str
        ``"html"`` or ``"pdf"``.
    email : str
        If provided, the report is emailed to this address after generation.
        Requires SMTP env vars to be configured.
    """
    results: dict[str, Any] = {}

    if format == "pdf":
        pub_result = _json.loads(publish_research_pdf(content, title))
    else:
        pub_result = _json.loads(publish_research_html(content, title))
    results["publish"] = pub_result

    if email.strip():
        filepath = pub_result.get("filepath", "")
        email_result = _json.loads(
            send_research_email(
                recipient=email.strip(),
                subject=f"FinAI Research: {title}",
                body=content,
                attachment_path=filepath,
            )
        )
        results["email"] = email_result

    return _json.dumps(results, indent=2)


YAHOO_FINANCE_TOOLS = [
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
    "publish_research_html": publish_research_html,
    "publish_research_pdf": publish_research_pdf,
    "publish_research_report": publish_research_report,
    "send_research_email": send_research_email,
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

