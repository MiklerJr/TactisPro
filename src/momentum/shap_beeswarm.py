"""Beeswarm of signed XGBoost contributions for the games already revealed."""

import pandas as pd
import plotly.graph_objects as go


def shap_beeswarm(contributions: pd.DataFrame) -> go.Figure:
    if contributions.empty:
        return go.Figure()

    selected = contributions.abs().mean().nlargest(5).index
    bound = float(contributions[selected].abs().max().max())
    spacing = max(bound, 0.01) * 0.035
    points = []
    for row, feature in enumerate(selected):
        placed = []
        for game, value in contributions[feature].dropna().sort_values().items():
            nearby = {lane for previous, lane in placed if abs(previous - value) < spacing}
            lane = next(
                lane for level in range(len(nearby) + 1)
                for lane in ([0] if level == 0 else [level, -level])
                if lane not in nearby
            )
            placed.append((value, lane))
            points.append((float(value), row, lane, feature, int(game)))

    if not points:
        return go.Figure()
    maximum = max(abs(point[2]) for point in points)
    offset = min(0.115, 0.38 / max(1, maximum))
    x, y, colors, hover = [], [], [], []
    for value, row, lane, feature, game in points:
        x.append(value)
        y.append(row + lane * offset)
        colors.append("#087f73" if value > 0 else "#b95b2d" if value < 0 else "#64748b")
        hover.append([feature, game])

    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers", customdata=hover, marker=dict(size=6, color=colors, opacity=0.84),
        hovertemplate="%{customdata[0]} · Juego %{customdata[1]}<br>SHAP %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=340, margin=dict(l=24, r=24, t=18, b=45), paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff", font=dict(family="system-ui, sans-serif", color="#64748b", size=11),
        xaxis=dict(title="Contribución SHAP", gridcolor="#e2e8f0", zeroline=False,
                   range=[-max(bound, 0.01) * 1.1, max(bound, 0.01) * 1.1]),
        yaxis=dict(tickmode="array", tickvals=list(range(len(selected))), ticktext=list(selected),
                   range=[len(selected) - 0.55, -0.45], showgrid=False, automargin=True),
    )
    fig.add_vline(x=0, line=dict(color="#64748b", dash="dot", width=1))
    return fig
