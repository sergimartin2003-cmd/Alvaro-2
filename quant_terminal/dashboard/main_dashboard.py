"""Dashboard web en tiempo real (Dash/Plotly, dark mode).

Dash y Plotly se importan de forma perezosa en ``build`` para no acoplar el
núcleo a ellos. Los helpers de figuras también importan plotly localmente.
"""

from __future__ import annotations

_DARK_BG = "#0e1117"
_PANEL_BG = "#161b22"
_ACCENT = "#6bcf7f"

_OPP_COLUMNS = [
    {"name": "Symbol", "id": "symbol"},
    {"name": "Score", "id": "score"},
    {"name": "Signal", "id": "signal"},
    {"name": "Price", "id": "price"},
    {"name": "Exp 24h", "id": "expected_24h"},
    {"name": "R/R", "id": "risk_reward"},
    {"name": "Entry", "id": "entry"},
    {"name": "Stop", "id": "stop_loss"},
    {"name": "TP2", "id": "take_profit_2"},
]


class TradingDashboard:
    """Dashboard con overview de mercado, oportunidades, riesgos y detalle."""

    def __init__(self, ranking_engine, telegram_alerts=None, refresh_ms: int = 30000) -> None:
        self.ranking_engine = ranking_engine
        self.telegram_alerts = telegram_alerts
        self.refresh_ms = refresh_ms
        self.app = None

    # ------------------------------------------------------------- layout
    def build(self):
        import dash
        from dash import dash_table, dcc, html

        self.app = dash.Dash(__name__, title="Quant Trading Terminal")
        cell_style = {"textAlign": "center", "backgroundColor": _PANEL_BG, "color": "#e6edf3",
                      "border": "1px solid #30363d"}
        conditional = [
            {"if": {"filter_query": '{signal} = "STRONG_BUY"', "column_id": "signal"},
             "backgroundColor": "#1a7f37", "color": "white", "fontWeight": "bold"},
            {"if": {"filter_query": '{signal} = "BUY"', "column_id": "signal"},
             "backgroundColor": "#238636", "color": "white"},
            {"if": {"filter_query": '{signal} = "STRONG_SELL"', "column_id": "signal"},
             "backgroundColor": "#da3633", "color": "white", "fontWeight": "bold"},
            {"if": {"filter_query": '{signal} = "SELL"', "column_id": "signal"},
             "backgroundColor": "#b62324", "color": "white"},
        ]

        self.app.layout = html.Div(
            style={"backgroundColor": _DARK_BG, "color": "#e6edf3", "fontFamily": "Inter, sans-serif",
                   "padding": "16px", "minHeight": "100vh"},
            children=[
                html.Div(
                    [
                        html.H1("🚀 Quantitative Trading Terminal", style={"margin": 0}),
                        html.Div(
                            [
                                html.Span("Status: "),
                                html.Span("ONLINE", id="system-status", style={"color": _ACCENT}),
                                html.Span(" | Last Update: "),
                                html.Span(id="last-update-time"),
                                html.Span(" | Assets: "),
                                html.Span(id="assets-count"),
                            ],
                            style={"opacity": 0.8},
                        ),
                    ]
                ),
                html.H3("Market Overview"),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"},
                    children=[
                        dcc.Graph(id="market-sentiment-gauge"),
                        dcc.Graph(id="asset-distribution-pie"),
                        dcc.Graph(id="class-scores-bar"),
                    ],
                ),
                html.Div(id="market-regime-indicator", style={"fontSize": "20px", "margin": "8px 0"}),
                html.H3("🔥 Top Opportunities"),
                dash_table.DataTable(
                    id="top-opportunities-table", columns=_OPP_COLUMNS,
                    style_cell=cell_style, style_data_conditional=conditional,
                    style_header={"backgroundColor": "#21262d", "fontWeight": "bold"},
                    row_selectable="single", selected_rows=[],
                ),
                html.Div(id="asset-detail-panel", style={"marginTop": "12px"}),
                html.H3("⚠️ Top Risks"),
                dash_table.DataTable(
                    id="top-risks-table", columns=_OPP_COLUMNS,
                    style_cell=cell_style, style_data_conditional=conditional,
                    style_header={"backgroundColor": "#21262d", "fontWeight": "bold"},
                ),
                html.H3("📊 Ranking by Asset Class"),
                dcc.Tabs(
                    id="class-tabs", value="stocks",
                    children=[dcc.Tab(label=c.title(), value=c)
                              for c in ("stocks", "forex", "crypto", "commodities", "indices")],
                ),
                html.Div(id="class-ranking-content"),
                dcc.Interval(id="interval-component", interval=self.refresh_ms, n_intervals=0),
            ],
        )
        self._register_callbacks()
        return self.app

    # ---------------------------------------------------------- callbacks
    def _register_callbacks(self):
        from dash import dash_table, html
        from dash.dependencies import Input, Output

        @self.app.callback(
            [Output("market-sentiment-gauge", "figure"),
             Output("asset-distribution-pie", "figure"),
             Output("class-scores-bar", "figure"),
             Output("market-regime-indicator", "children"),
             Output("top-opportunities-table", "data"),
             Output("top-risks-table", "data"),
             Output("last-update-time", "children"),
             Output("assets-count", "children")],
            [Input("interval-component", "n_intervals")],
        )
        def _update(_n):
            summary = self.ranking_engine.get_market_summary()
            opps = [a.to_row() for a in self.ranking_engine.get_top_opportunities(10, min_score=0)]
            risks = [a.to_row() for a in self.ranking_engine.get_top_risks(10, max_score=100)]
            regime = f"Market Regime: {summary.get('market_sentiment', 'NEUTRAL')}"
            ts = summary.get("timestamp")
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"
            return (
                self.create_sentiment_gauge(summary.get("average_score", 50)),
                self.create_distribution_pie(summary),
                self.create_class_scores_bar(summary),
                regime,
                opps,
                risks,
                ts_str,
                summary.get("total_assets_analyzed", 0),
            )

        @self.app.callback(
            Output("asset-detail-panel", "children"),
            [Input("top-opportunities-table", "selected_rows"),
             Input("top-opportunities-table", "data")],
        )
        def _detail(selected_rows, data):
            if not selected_rows or not data:
                return None
            symbol = data[selected_rows[0]]["symbol"]
            analysis = self.ranking_engine.current_rankings.get(symbol)
            if analysis is None:
                return None
            return self.create_asset_detail_content(analysis)

        @self.app.callback(
            Output("class-ranking-content", "children"),
            [Input("class-tabs", "value")],
        )
        def _class(tab):
            rows = [a.to_row() for a in self.ranking_engine.get_ranking_by_class(tab)]
            return dash_table.DataTable(
                columns=_OPP_COLUMNS, data=rows,
                style_cell={"textAlign": "center", "backgroundColor": _PANEL_BG, "color": "#e6edf3"},
            )

    # ----------------------------------------------------------- figuras
    @staticmethod
    def create_sentiment_gauge(score: float):
        import plotly.graph_objs as go

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number", value=score, title={"text": "Market Sentiment"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#58a6ff"},
                    "steps": [
                        {"range": [0, 30], "color": "#ff6b6b"},
                        {"range": [30, 70], "color": "#ffd93d"},
                        {"range": [70, 100], "color": "#6bcf7f"},
                    ],
                },
            )
        )
        fig.update_layout(height=250, paper_bgcolor=_DARK_BG, font_color="#e6edf3")
        return fig

    @staticmethod
    def create_distribution_pie(summary: dict):
        import plotly.graph_objs as go

        values = [summary.get("bullish_assets", 0), summary.get("bearish_assets", 0),
                  summary.get("neutral_assets", 0)]
        fig = go.Figure(
            go.Pie(labels=["Bullish", "Bearish", "Neutral"], values=values,
                   marker={"colors": ["#6bcf7f", "#ff6b6b", "#ffd93d"]})
        )
        fig.update_layout(height=250, paper_bgcolor=_DARK_BG, font_color="#e6edf3")
        return fig

    @staticmethod
    def create_class_scores_bar(summary: dict):
        import plotly.graph_objs as go

        by_class = summary.get("by_asset_class", {})
        classes = list(by_class.keys())
        scores = [by_class[c]["avg_score"] for c in classes]
        fig = go.Figure(go.Bar(x=classes, y=scores, marker_color="#58a6ff"))
        fig.update_layout(height=250, paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
                          font_color="#e6edf3", yaxis={"range": [0, 100]})
        return fig

    def create_asset_detail_content(self, analysis):
        from dash import dcc, html

        radar = self._radar(analysis)
        return html.Div(
            style={"backgroundColor": _PANEL_BG, "padding": "16px", "borderRadius": "8px"},
            children=[
                html.H4(f"{analysis.symbol} — {analysis.action} ({analysis.final_score:.1f}/100)"),
                dcc.Graph(figure=radar),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                    children=[
                        html.Div([html.B("Razones para comprar"),
                                  html.Ul([html.Li(r) for r in analysis.reasons_to_buy])]),
                        html.Div([html.B("Razones para evitar"),
                                  html.Ul([html.Li(r) for r in analysis.reasons_to_avoid])]),
                    ],
                ),
                html.Div([html.B("Catalizadores"), html.Ul([html.Li(c) for c in analysis.key_catalysts])]),
                html.Div([html.B("Riesgos"), html.Ul([html.Li(r) for r in analysis.risk_factors])]),
            ],
        )

    @staticmethod
    def _radar(analysis):
        import plotly.graph_objs as go

        cats = ["technical", "sentiment", "macro", "options_flow", "ml_prediction",
                "seasonality", "volume_profile", "correlation"]
        vals = [getattr(analysis, f"{c}_score") for c in cats]
        fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                                        line_color="#58a6ff"))
        fig.update_layout(height=300, paper_bgcolor=_PANEL_BG, font_color="#e6edf3",
                          polar={"radialaxis": {"range": [0, 100]}})
        return fig

    def run(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
        if self.app is None:
            self.build()
        self.app.run(host=host, port=port, debug=debug)
