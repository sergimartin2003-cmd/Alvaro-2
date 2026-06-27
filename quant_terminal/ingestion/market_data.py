"""Cliente de datos de mercado (Polygon / Alpaca), importados de forma perezosa."""

from __future__ import annotations

import pandas as pd


class MarketDataClient:
    def __init__(self, polygon_api_key: str | None = None,
                 alpaca_api_key: str | None = None, alpaca_secret: str | None = None,
                 data_url: str = "wss://stream.data.alpaca.markets/v2/sip") -> None:
        self.polygon_api_key = polygon_api_key
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret = alpaca_secret
        self.data_url = data_url
        self._alpaca_rest = None

    @property
    def alpaca_rest(self):
        if self._alpaca_rest is None:
            from alpaca_trade_api import REST

            self._alpaca_rest = REST(self.alpaca_api_key, self.alpaca_secret)
        return self._alpaca_rest

    def get_historical_data(self, symbol: str, timeframe: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
        """Descarga barras OHLCV históricas vía Alpaca.

        timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'.
        """
        bars = self.alpaca_rest.get_bars(symbol, timeframe, start=start_date, end=end_date).df
        bars = bars.rename(
            columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        )
        return bars[["open", "high", "low", "close", "volume"]]

    def get_options_chain(self, symbol: str, expiry_date: str | None = None) -> dict:
        """Cadena de opciones vía Polygon (requiere polygon-api-client)."""
        from polygon import RESTClient

        client = RESTClient(self.polygon_api_key)
        calls, puts = [], []
        for c in client.list_options_contracts(underlying_ticker=symbol, expiration_date=expiry_date):
            entry = {"strike": c.strike_price, "expiry": c.expiration_date, "type": c.contract_type}
            (calls if c.contract_type == "call" else puts).append(entry)
        return {"calls": calls, "puts": puts}

    async def stream_quotes(self, symbols: list[str], on_quote):
        """Stream de quotes en tiempo real vía WebSocket de Alpaca."""
        from alpaca_trade_api import Stream

        stream = Stream(self.alpaca_api_key, self.alpaca_secret, data_url=self.data_url)

        async def _handler(q):
            await on_quote(q)

        for sym in symbols:
            stream.subscribe_quotes(_handler, sym)
        stream.run()
