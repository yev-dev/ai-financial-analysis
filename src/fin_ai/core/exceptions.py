"""
Custom exceptions for the FinAI data services.
"""


class MarketDataNotFoundError(LookupError):
    """Raised when requested market data is not available.

    This can happen in two scenarios:

    * **Offline mode** — the serialised data file for the requested ticker
      and data type does not exist in ``data/<SYMBOL>/``.
    * **Online mode** — the yfinance API returned an empty result or the
      ticker does not exist.
    """

    def __init__(
        self,
        symbol: str,
        data_type: str,
        detail: str = "",
    ) -> None:
        self.symbol = symbol
        self.data_type = data_type
        self.detail = detail
        message = (
            f"No {data_type} data available for '{symbol}'."
            f"{'  ' + detail if detail else ''}"
        )
        super().__init__(message)


class MarketDataServiceError(RuntimeError):
    """Generic error from the market data service layer.

    Wraps lower-level I/O, serialisation, or API errors so callers can
    catch a single exception type if they do not need granularity.
    """

    def __init__(
        self,
        message: str,
        symbol: str = "",
        data_type: str = "",
        cause: Exception | None = None,
    ) -> None:
        self.symbol = symbol
        self.data_type = data_type
        self.cause = cause
        full = message
        if symbol:
            full += f"  [symbol={symbol}]"
        if data_type:
            full += f"  [type={data_type}]"
        super().__init__(full)
