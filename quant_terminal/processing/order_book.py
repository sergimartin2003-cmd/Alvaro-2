"""Análisis de imbalance del order book (PROMPT 1.2).

Detecta imbalance bid/ask, vacíos de liquidez, spoofing, iceberg orders y
puntúa la profundidad. Solo numpy/collections.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class OrderBookImbalanceAnalyzer:
    def __init__(self, max_history_size: int = 10000) -> None:
        self.history: deque = deque(maxlen=max_history_size)

    @staticmethod
    def _vol(levels):
        return float(sum(v for _p, v in levels))

    def calculate_bid_ask_imbalance(self, order_book: dict, levels: int = 10) -> dict:
        bids = order_book["bids"][:levels]
        asks = order_book["asks"][:levels]
        bid_vol = self._vol(bids)
        ask_vol = self._vol(asks)
        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total if total else 0.0

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0.0

        # Imbalance ponderado por cercanía al mid (exp decay).
        w_bid = sum(v * np.exp(-abs(p - mid) / mid) for p, v in bids) if mid else bid_vol
        w_ask = sum(v * np.exp(-abs(p - mid) / mid) for p, v in asks) if mid else ask_vol
        w_total = w_bid + w_ask
        weighted = (w_bid - w_ask) / w_total if w_total else 0.0

        if imbalance > 0.5:
            signal = "STRONG_BUY"
        elif imbalance > 0.3:
            signal = "BUY"
        elif imbalance < -0.5:
            signal = "STRONG_SELL"
        elif imbalance < -0.3:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        spread_bps = ((best_ask - best_bid) / mid * 10000) if mid else 0.0
        self.history.append(imbalance)
        return {
            "imbalance": float(imbalance),
            "bid_volume": bid_vol,
            "ask_volume": ask_vol,
            "weighted_imbalance": float(weighted),
            "signal": signal,
            "confidence": float(min(1.0, abs(imbalance) * 1.5)),
            "mid_price": float(mid),
            "spread_bps": float(spread_bps),
        }

    def detect_liquidity_voids(self, order_book: dict, threshold_pct: float = 0.1) -> list[dict]:
        out = []
        for side in ("bids", "asks"):
            levels = order_book[side]
            if len(levels) < 2:
                continue
            vols = np.array([v for _p, v in levels], dtype=float)
            avg = vols.mean()
            for i in range(len(levels) - 1):
                if levels[i][1] < threshold_pct * avg:
                    p0, p1 = levels[i][0], levels[i + 1][0]
                    out.append({
                        "type": "BID_VOID" if side == "bids" else "ASK_VOID",
                        "price_range": (float(min(p0, p1)), float(max(p0, p1))),
                        "void_severity": float(1 - levels[i][1] / avg) if avg else 0.0,
                        "estimated_slippage": float(abs(p1 - p0)),
                        "trading_implication": "Posible movimiento rápido a través del vacío",
                    })
        return out

    def detect_spoofing(self, symbol, order_book_updates, window_ms: int = 1000) -> list[dict]:
        """Órdenes grandes canceladas rápido sin ejecutarse."""
        out = []
        vols = [u.get("volume", 0) for u in order_book_updates]
        avg = np.mean(vols) if vols else 0
        for u in order_book_updates:
            if (
                u.get("volume", 0) > 5 * avg
                and u.get("action") == "cancel"
                and u.get("lifetime_ms", 1e9) < window_ms
                and not u.get("executed", False)
            ):
                out.append({
                    "side": "BID_SIDE" if u.get("side") == "bid" else "ASK_SIDE",
                    "price_level": float(u.get("price", 0)),
                    "spoofed_volume": float(u.get("volume", 0)),
                    "duration_ms": int(u.get("lifetime_ms", 0)),
                    "confidence": float(min(1.0, u["volume"] / (10 * avg))) if avg else 0.5,
                    "market_impact": float(u.get("volume", 0) / (avg + 1e-9)),
                })
        return out

    def detect_iceberg_orders(self, symbol, trades, order_book, window_seconds: int = 60) -> list[dict]:
        """Múltiples fills del mismo tamaño/precio con replenishment."""
        from collections import defaultdict

        groups: dict[tuple, list] = defaultdict(list)
        for t in trades:
            groups[(round(t["price"], 4), t.get("size"))].append(t)
        out = []
        for (price, size), fills in groups.items():
            if size and len(fills) >= 4:
                out.append({
                    "side": fills[0].get("side", "BUY"),
                    "price_level": float(price),
                    "estimated_total_size": float(size * len(fills)),
                    "confidence": float(min(1.0, len(fills) / 10)),
                    "institutional_activity": len(fills) >= 6,
                })
        return out

    def calculate_market_depth_score(self, order_book: dict) -> dict:
        bids, asks = order_book["bids"], order_book["asks"]
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0
        spread_bps = ((best_ask - best_bid) / mid * 10000) if mid else 100
        top5 = self._vol(bids[:5]) + self._vol(asks[:5])
        stability = float(np.clip(1 - np.std(self.history) if self.history else 0.5, 0, 1))
        score = float(np.clip(100 - spread_bps * 2 + min(top5 / 1000, 30) + stability * 20, 0, 100))
        grade = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D" if score >= 20 else "F"
        return {
            "depth_score": score,
            "spread_bps": float(spread_bps),
            "top5_volume": float(top5),
            "stability": stability,
            "liquidity_grade": grade,
        }
