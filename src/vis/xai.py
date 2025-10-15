from typing import Dict, Optional, List, Tuple

import numpy as np
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

def visualize_sae_features_3d(
    feature_stats: Dict[str, np.ndarray],
    num_features: Optional[int] = None,
    ev: Optional[float] = None,
    top_k: int = 20,
    alpha_de: float = 0.6,
    density_target_log10: float = -2.5,
    density_sigma: float = 0.5
) -> Tuple[go.Figure, Dict[str, np.ndarray]]:
    """
    Create 3D visualization of SAE features with interpretability scoring.

    Generates an interactive 3D scatter plot showing features along three axes:
    encoder-decoder cosine similarity, log10 density, and encoder bias. Features
    are colored by interpretability score and sized by reconstruction contribution.

    Parameters
    ----------
    feature_stats : Dict[str, np.ndarray]
        Dictionary of feature statistics from _compute_feature_statistics
    num_features : Optional[int], optional
        Number of features to subsample for visualization, by default None (use all)
    ev : Optional[float], optional
        Explained variance value for title, by default None
    top_k : int, optional
        Number of top features to highlight, by default 20
    alpha_de : float, optional
        Weight of D/E cosine in interpretability score (0-1), by default 0.6
    density_target_log10 : float, optional
        Target log10 density for interpretability scoring, by default -2.5
    density_sigma : float, optional
        Width of Gaussian for density scoring in log10 space, by default 0.5

    Returns
    -------
    Tuple[go.Figure, Dict[str, np.ndarray]]
        Plotly Figure object and extended statistics dictionary

    Examples
    --------
    >>> stats = _compute_feature_statistics(model, activations)
    >>> fig, extended_stats = visualize_sae_features_3d(stats, top_k=10)
    >>> fig.show()
    """
    feature_stats = feature_stats.copy()
    p = feature_stats['density'].shape[0]

    # Subsample features if requested
    if (num_features is not None) and (num_features < p):
        idx = np.random.choice(p, size=num_features, replace=False)
        for key in feature_stats:
            feature_stats[key] = feature_stats[key][idx]

    # Extract base statistics
    density = feature_stats['density']
    density_log10 = np.log10(np.clip(density, 1e-12, None))
    de_cos = feature_stats['de_cosine']
    de_frac = feature_stats['de_frac_energy']
    proj_coeff = feature_stats['de_proj_coeff']
    raw_dot = feature_stats['de_dot']
    bias = feature_stats['bias']
    recon_contrib = feature_stats['recon_contrib']

    # Compute interpretability score combining D/E cosine and density proximity
    de_norm = (de_cos + 1.0) / 2.0  # Normalize cosine from [-1,1] to [0,1]
    dist = np.abs(density_log10 - density_target_log10)
    density_score = np.exp(-0.5 * (dist / max(1e-6, density_sigma))**2)
    alpha_de = float(np.clip(alpha_de, 0.0, 1.0))
    interpret_raw = alpha_de * de_norm + (1.0 - alpha_de) * density_score
    interp_min, interp_max = interpret_raw.min(), interpret_raw.max()
    interpretability = (interpret_raw - interp_min) / (max(1e-12, interp_max - interp_min))

    # Scale marker sizes by reconstruction contribution
    recon_norm = recon_contrib / (recon_contrib.max() + 1e-12)
    marker_sizes = 6.0 + 38.0 * np.sqrt(recon_norm)
    marker_sizes = np.clip(marker_sizes, 3.0, 60.0)

    # Create density masks for categorization
    low_thresh, high_thresh = -4.0, -1.0
    low_mask = density_log10 < low_thresh
    mid_mask = (density_log10 >= low_thresh) & (density_log10 < high_thresh)
    high_mask = density_log10 >= high_thresh

    # Identify top features by reconstruction contribution
    top_k = int(min(top_k, len(recon_contrib)))
    top_idx = np.argsort(recon_contrib)[-top_k:]

    fig = go.Figure()

    # Low density features trace with colorbar
    fig.add_trace(go.Scatter3d(
        x=de_cos[low_mask],
        y=density_log10[low_mask],
        z=bias[low_mask],
        mode='markers',
        marker=dict(
            size=marker_sizes[low_mask],
            color=interpretability[low_mask],
            colorscale='Turbo',
            cmin=0.0, cmax=1.0,
            colorbar=dict(title='Interpretability', x=0.92),
            opacity=0.7,
            line=dict(width=0)
        ),
        text=[
            f"feat={i}<br>interp={interpretability[i]:.3f}<br>"
            f"D/E cos={de_cos[i]:.3f} (frac={de_frac[i]:.3f})<br>"
            f"proj_coeff={proj_coeff[i]:.3f}<br>"
            f"density={density[i]:.4f} (log10={density_log10[i]:.3f})<br>"
            f"bias={bias[i]:.4f}<br>recon_frac={recon_contrib[i]:.4f}"
            for i in np.where(low_mask)[0]
        ],
        hovertemplate='%{text}<extra></extra>',
        name=f'Low Density (log10 < {low_thresh})'
    ))

    # Medium density features
    fig.add_trace(go.Scatter3d(
        x=de_cos[mid_mask],
        y=density_log10[mid_mask],
        z=bias[mid_mask],
        mode='markers',
        marker=dict(
            size=marker_sizes[mid_mask],
            color=interpretability[mid_mask],
            colorscale='Turbo',
            showscale=False,
            opacity=0.85,
            line=dict(width=0)
        ),
        text=[
            f"feat={i}<br>interp={interpretability[i]:.3f}<br>"
            f"D/E cos={de_cos[i]:.3f} (frac={de_frac[i]:.3f})<br>"
            f"proj_coeff={proj_coeff[i]:.3f}<br>"
            f"density={density[i]:.4f} (log10={density_log10[i]:.3f})<br>"
            f"bias={bias[i]:.4f}<br>recon_frac={recon_contrib[i]:.4f}"
            for i in np.where(mid_mask)[0]
        ],
        hovertemplate='%{text}<extra></extra>',
        name='Medium Density'
    ))

    # High density features
    fig.add_trace(go.Scatter3d(
        x=de_cos[high_mask],
        y=density_log10[high_mask],
        z=bias[high_mask],
        mode='markers',
        marker=dict(
            size=marker_sizes[high_mask],
            color=interpretability[high_mask],
            colorscale='Turbo',
            showscale=False,
            opacity=0.95,
            line=dict(width=0)
        ),
        text=[
            f"feat={i}<br>interp={interpretability[i]:.3f}<br>"
            f"D/E cos={de_cos[i]:.3f} (frac={de_frac[i]:.3f})<br>"
            f"proj_coeff={proj_coeff[i]:.3f}<br>"
            f"density={density[i]:.4f} (log10={density_log10[i]:.3f})<br>"
            f"bias={bias[i]:.4f}<br>recon_frac={recon_contrib[i]:.4f}"
            for i in np.where(high_mask)[0]
        ],
        hovertemplate='%{text}<extra></extra>',
        name=f'High Density (log10 ≥ {high_thresh})'
    ))

    # Top features overlay with diamond markers
    fig.add_trace(go.Scatter3d(
        x=de_cos[top_idx],
        y=density_log10[top_idx],
        z=bias[top_idx],
        mode='markers+text',
        marker=dict(
            size=(marker_sizes[top_idx] * 1.6),
            color=interpretability[top_idx],
            colorscale='Turbo',
            showscale=False,
            opacity=1.0,
            symbol='diamond',
            line=dict(width=5, color='white')
        ),
        text=[f"#{i}" for i in top_idx],
        textposition="top center",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "interp=%{marker.color:.3f}<br>"
            "D/E cos=%{x:.3f}<br>"
            "density=%{y:.4f} (log10=%{y:.3f})<br>"
            "bias=%{z:.4f}<extra></extra>"
        ),
        name=f'Top {top_k}'
    ))

    # Configure layout and title
    title = '3D SAE features — X: D/E (cos) | Y: log10(density) | Z: encoder bias'
    if ev is not None:
        title += f' — model EV={ev:.3f}'

    fig.update_layout(
        title=title,
        template='plotly_dark',
        scene=dict(
            xaxis=dict(title='D/E (cosine)'),
            yaxis=dict(title='log10(Feature density)'),
            zaxis=dict(title='Encoder bias'),
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1.0, z=0.6),
            camera=dict(eye=dict(x=0.9, y=0.9, z=0.6))
        ),
        width=1800,
        height=1200,
        legend=dict(x=0.02, y=0.98)
    )

    # Prepare extended statistics output
    out_stats = {
        'density': density,
        'density_log10': density_log10,
        'bias': bias,
        'de_cosine': de_cos,
        'de_frac_energy': de_frac,
        'de_proj_coeff': proj_coeff,
        'de_dot': raw_dot,
        'recon_contrib': recon_contrib,
        'interpretability': interpretability,
        'marker_sizes': marker_sizes,
    }

    return fig, out_stats
