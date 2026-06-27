"""Dashboard web en tiempo real (Dash/Plotly).

Dash y Plotly se importan de forma perezosa al construir el dashboard, de modo
que el resto del paquete no depende de ellos.
"""

from __future__ import annotations


class RealTimeDashboard:
    """Dashboard con auto-refresh para señales, régimen y alertas."""

    def __init__(self, signal_engine, alert_system, refresh_ms: int = 5000) -> None:
        self.signal_engine = signal_engine
        self.alert_system = alert_system
        self.refresh_ms = refresh_ms
        self.app = None

    def build(self):
        import dash
        from dash import dcc, html

        self.app = dash.Dash(__name__)
        self.app.layout = html.Div(
            [
                html.H1("Quantitative Trading Terminal", style={"textAlign": "center"}),
                html.Div(id="active-signals"),
                dcc.Graph(id="signal-heatmap"),
                html.Div(id="recent-alerts"),
                dcc.Interval(id="interval", interval=self.refresh_ms, n_intervals=0),
            ]
        )
        self._register_callbacks()
        return self.app

    def _register_callbacks(self) -> None:
        from dash import html
        from dash.dependencies import Input, Output

        @self.app.callback(
            [Output("active-signals", "children"), Output("recent-alerts", "children")],
            [Input("interval", "n_intervals")],
        )
        def _update(_n):
            signals = getattr(self.signal_engine, "get_active_signals", lambda: [])()
            alerts = self.alert_system.alert_history[-10:]
            sig_html = html.Ul([html.Li(str(s)) for s in signals])
            alert_html = html.Ul([html.Li(str(a["alert"].get("title"))) for a in alerts])
            return sig_html, alert_html

    def run(self, host: str = "0.0.0.0", port: int = 8050) -> None:
        if self.app is None:
            self.build()
        self.app.run(host=host, port=port, debug=False)
