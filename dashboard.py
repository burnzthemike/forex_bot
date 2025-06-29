# dashboard.py

import threading
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from config import DASHBOARD_PORT, DASHBOARD_REFRESH

# Shared state populated by your engine
shared_data = {
    "equity_curve": [],
    "latest_metrics": {}
}

def start_dashboard():
    app = Dash(__name__)
    app.title = "Forex Bot Dashboard"

    app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'margin': '20px'}, children=[
        html.H1("📈 Forex Bot Live Dashboard"),
        html.Div(id="stats", style={'marginBottom': '20px'}),
        dcc.Graph(id="equity-graph", config={'displayModeBar': False}),
        dcc.Interval(id="interval", interval=DASHBOARD_REFRESH * 1000, n_intervals=0)
    ])

    @app.callback(
        Output("equity-graph", "figure"),
        Output("stats", "children"),
        Input("interval", "n_intervals")
    )
    def update_dashboard(n):
        curve = shared_data.get("equity_curve", [])
        metrics = shared_data.get("latest_metrics", {})

        # Build equity-curve figure
        if curve:
            fig = go.Figure(
                go.Scatter(
                    y=curve,
                    mode="lines",
                    name="Equity"
                )
            )
            fig.update_layout(
                xaxis_title="Trade Number",
                yaxis_title="Equity",
                margin=dict(l=40, r=20, t=40, b=40)
            )
        else:
            # Empty placeholder
            fig = go.Figure()
            fig.update_layout(
                title="No equity data yet",
                margin=dict(l=40, r=20, t=40, b=40)
            )

        # Build stats panel
        if metrics:
            stats_children = [
                html.Div(f"{key}: {value:.4f}", style={'padding': '4px 0'})
                for key, value in metrics.items()
            ]
        else:
            stats_children = [html.Div("Waiting for data...", style={'fontStyle': 'italic'})]

        return fig, stats_children

    # Run Dash in its own thread so it doesn’t block your engine
    thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=DASHBOARD_PORT, debug=False),
        daemon=True
    )
    thread.start()
