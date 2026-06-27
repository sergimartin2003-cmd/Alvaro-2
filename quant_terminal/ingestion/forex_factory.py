"""Scraper del calendario económico de Forex Factory.

Usa requests + BeautifulSoup (importados de forma perezosa). El parsing de FF
cambia con frecuencia; este módulo aísla esa lógica para facilitar su
mantenimiento.
"""

from __future__ import annotations

import pandas as pd

_IMPACT_MAP = {
    "icon--ff-impact-red": "High",
    "icon--ff-impact-ora": "Medium",
    "icon--ff-impact-yel": "Low",
}


class ForexFactoryScraper:
    def __init__(self) -> None:
        self.base_url = "https://www.forexfactory.com/calendar"
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def get_events(self, date_range: str = "this_week") -> pd.DataFrame:
        """Descarga y parsea eventos. Devuelve un DataFrame con columnas
        estándar: date, time, currency, event, impact, forecast, previous, actual.
        """
        import requests
        from bs4 import BeautifulSoup

        url = f"{self.base_url}?week={date_range}"
        resp = requests.get(url, headers=self.headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        events = []
        for row in soup.select("tr.calendar__row"):
            currency = row.select_one(".calendar__currency")
            event = row.select_one(".calendar__event")
            if not event:
                continue
            impact_el = row.select_one(".calendar__impact span")
            impact = "Low"
            if impact_el:
                for cls, lvl in _IMPACT_MAP.items():
                    if cls in (impact_el.get("class") or []):
                        impact = lvl
            events.append(
                {
                    "time": self._text(row.select_one(".calendar__time")),
                    "currency": self._text(currency),
                    "event": self._text(event),
                    "impact": impact,
                    "forecast": self._num(row.select_one(".calendar__forecast")),
                    "previous": self._num(row.select_one(".calendar__previous")),
                    "actual": self._num(row.select_one(".calendar__actual")),
                }
            )
        return pd.DataFrame(events)

    @staticmethod
    def _text(el) -> str:
        return el.get_text(strip=True) if el else ""

    @staticmethod
    def _num(el):
        if not el:
            return None
        raw = el.get_text(strip=True).replace("%", "").replace(",", "").replace("K", "e3").replace("M", "e6")
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def filter_high_impact(events_df: pd.DataFrame) -> pd.DataFrame:
        return events_df[events_df["impact"] == "High"]

    @staticmethod
    def calculate_surprise_factor(event: dict, std_dev: float = 1.0) -> float:
        actual, forecast = event.get("actual"), event.get("forecast")
        if actual is None or forecast is None:
            return 0.0
        return (actual - forecast) / (std_dev or 1.0)
