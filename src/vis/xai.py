from typing import List, Tuple

import plotly.graph_objects as go


def visualize_top_tokens(
    token_scores: List[Tuple[str, float]],
    title: str = "Top Tokens by Relevance",
    color_scale: str = "Viridis",
    height: int = 900,
    width: int = 1600
) -> go.Figure:
    """
    Create an interactive horizontal bar chart visualizing token relevance scores.

    Generates a Plotly figure displaying tokens sorted by their relevance scores,
    with color intensity representing score magnitude.

    Parameters
    ----------
    token_scores : List[Tuple[str, float]]
        List of (token, score) tuples as returned by get_top_tokens()
    title : str, optional
        Chart title, by default "Top Tokens by Relevance"
    color_scale : str, optional
        Plotly colorscale name ('Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'),
        by default "Viridis"
    height : int, optional
        Figure height in pixels, by default 900
    width : int, optional
        Figure width in pixels, by default 1600

    Returns
    -------
    go.Figure
        Plotly Figure object with the token visualization

    Raises
    ------
    ValueError
        If token_scores is empty

    Examples
    --------
    >>> token_scores = [("important", 0.95), ("relevant", 0.82), ("test", 0.65)]
    >>> fig = visualize_top_tokens(token_scores, title="Top Relevant Tokens")
    >>> fig.show()
    """
    if not token_scores:
        raise ValueError("token_scores cannot be empty")

    tokens, scores = zip(*token_scores)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=tokens,
        x=scores,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale=color_scale,
            cmin=0,
            cmax=max(scores),
            colorbar=dict(title="Relevance Score"),
            line=dict(width=0)
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Relevance: %{x:.4f}<br>"
            "<extra></extra>"
        ),
        text=[f"{score:.3f}" for score in scores],
        textposition='auto',
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Relevance Score",
            range=[0, max(scores) * 1.05],
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            title="Tokens",
            tickfont=dict(size=12),
            categoryorder='total ascending'
        ),
        height=height,
        width=width,
        margin=dict(l=100, r=50, t=80, b=50),
        showlegend=False,
    )

    return fig
