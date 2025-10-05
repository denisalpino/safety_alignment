"""
📊 Prompt Data Visualization Module

🎯 Purpose:
    Create comprehensive interactive visualizations for analyzing prompt characteristics
    in datasets with clustering and data source support

✨ Advantages:
    • Interactive dark theme charts
    • Combined visualizations (histograms + KDE + box plots)
    • Heatmaps for multidimensional analysis
    • Grouped charts for category comparison
    • Automatic statistical annotations

🛠 Core Capabilities:
    1. Prompt length distribution analysis
    2. Average length comparison across sources and clusters
    3. Data distribution visualization between categories
"""

from typing import Tuple, Dict, Any

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


def create_prompt_length_distribution(dataset: pd.DataFrame) -> go.Figure:
    """
    Creates a combined visualization of prompt length distribution.

    Includes frequency histogram, density estimation (KDE), statistical markers,
    and box plot for comprehensive analysis of text prompt length distribution.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset with 'prompt' column containing text prompts

    Returns
    -------
    go.Figure
        Interactive Plotly figure with combined visualization
    """
    dataset['prompt_length'] = dataset['prompt'].str.len()
    data = dataset['prompt_length']

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.3],
        vertical_spacing=0.1,
        subplot_titles=(
            '<b>Distribution of Prompt Lengths</b>',
            '<b>Statistical Summary</b>'
        )
    )

    hist, bin_edges = np.histogram(data, bins=150, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    kde = gaussian_kde(data)
    x_kde = np.linspace(max(0, data.min()), data.max(), 500)
    y_kde = kde(x_kde)

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist,
            name='Frequency',
            marker_color='rgba(100, 150, 255, 0.6)',
            marker_line_color='rgba(70, 130, 255, 0.8)',
            marker_line_width=1,
            opacity=0.7,
            hovertemplate='<b>Length: %{x:.0f} chars</b><br>Count: %{y:.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde * (hist.max() / y_kde.max()),
            name='Density Estimate',
            line=dict(color='#ff6464', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 100, 100, 0.2)',
            hovertemplate='<b>Length: %{x:.0f} chars</b><br>Density: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    mean_length = data.mean()
    fig.add_trace(
        go.Scatter(
            x=[mean_length, mean_length],
            y=[0, hist.max() * 1.05],
            mode='lines',
            name=f'Mean: {mean_length:.0f} chars',
            line=dict(color='#00ff88', width=2, dash='dash'),
            hovertemplate='<b>Mean length: %{x:.0f} characters</b><extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(
            x=data,
            name='Statistics',
            boxpoints="outliers",
            fillcolor='rgba(255, 100, 100, 0.3)',
            line_color='#ff6464',
            whiskerwidth=0.2,
            hovertemplate=(
                '<b>Statistical Summary</b><br>'
                'Min: %{min}<br>'
                'Q1: %{q1}<br>'
                'Median: %{median}<br>'
                'Q3: %{q3}<br>'
                'Max: %{max}<extra></extra>'
            )
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=dict(
            text='<b>Prompt Length Distribution Analysis</b>',
            x=0.5,
            font=dict(size=26, color='#ffffff'),
            xanchor='center',
            y=0.98
        ),
        template='plotly_dark',
        width=1600,
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(30, 30, 60, 0.7)',
            bordercolor='rgba(100, 100, 200, 0.5)',
            borderwidth=1,
            font=dict(size=12)
        ),
        paper_bgcolor='rgba(15, 15, 35, 0.9)',
        plot_bgcolor='rgba(20, 20, 40, 0.8)',
        font=dict(color='#e0e0ff', family='Arial'),
        margin=dict(t=120, b=80, l=80, r=80)
    )

    fig.update_xaxes(
        row=1, col=1,
        title_text='<b>Prompt Length (characters)</b>',
        range=[0, data.max() * 1.05],
        gridcolor='rgba(100, 100, 200, 0.2)',
        gridwidth=1,
        title_font=dict(size=14, color='#a0f0ff')
    )
    fig.update_yaxes(
        row=1, col=1,
        title_text='<b>Frequency (Count)</b>',
        gridcolor='rgba(100, 100, 200, 0.2)',
        gridwidth=1,
        title_font=dict(size=14, color='#a0f0ff')
    )

    fig.update_xaxes(
        row=2, col=1,
        title_text='<b>Prompt Length (characters)</b>',
        range=[0, data.max() * 1.05],
        gridcolor='rgba(100, 100, 200, 0.2)',
        gridwidth=1,
        title_font=dict(size=14, color='#a0f0ff')
    )
    fig.update_yaxes(
        row=2, col=1,
        showticklabels=False,
        showgrid=False
    )

    stats_text = f"""
    <b>Key Statistics:</b><br>
    • Total prompts: {len(data):,}<br>
    • Mean length: {mean_length:.0f} chars<br>
    • Median length: {data.median():.0f} chars<br>
    • Std deviation: {data.std():.0f} chars<br>
    • Range: {data.min():.0f} - {data.max():.0f} chars
    """

    fig.add_annotation(
        x=0.98, y=0.98,
        xref="paper", yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=11, color="#e0e0ff"),
        bgcolor="rgba(30, 30, 60, 0.8)",
        bordercolor="#4fa8ff",
        borderwidth=1,
        borderpad=8,
        align="right",
        xanchor="right"
    )

    return fig


def create_length_heatmap(dataset: pd.DataFrame) -> go.Figure:
    """
    Creates a heatmap of average prompt lengths by sources and clusters.

    Visualizes multidimensional relationships between data categories,
    helping to identify patterns in prompt length distribution.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset with 'source', 'cluster', 'prompt_length' columns

    Returns
    -------
    go.Figure
        Interactive heatmap with value annotations
    """
    pivot_data = dataset.pivot_table(
        index='source',
        columns='cluster',
        values='prompt_length',
        aggfunc='mean'
    ).fillna(0)

    annotations = []
    for i, source in enumerate(pivot_data.index):
        for j, cluster in enumerate(pivot_data.columns):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(int(pivot_data.iloc[i, j])),
                    font=dict(color='white', size=12, weight='bold'),
                    showarrow=False
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[col for col in pivot_data.columns],
        y=pivot_data.index,
        colorscale='Viridis',
        hoverongaps=False,
        hoverinfo='z+x+y',
        showscale=True,
        colorbar=dict(title="Avg Length"),
        textfont={"size": 12, "color": "white"}
    ))

    fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=dict(
            text='<b>Average Prompt Length by Source and Cluster</b>',
            x=0.5,
            font=dict(size=20, color='#ffffff')
        ),
        xaxis_title='<b>Clusters</b>',
        yaxis_title='<b>Data Sources</b>',
        template='plotly_dark',
        paper_bgcolor='rgba(15, 15, 35, 0.9)',
        plot_bgcolor='rgba(20, 20, 40, 0.8)',
        height=900,
        width=1600,
        font=dict(color='#e0e0ff')
    )

    return fig


def create_grouped_bar_chart(dataset: pd.DataFrame) -> go.Figure:
    """
    Creates a grouped bar chart of data distribution.

    Visualizes prompt counts by data sources and clusters,
    enabling comparison of category distribution across different groups.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset with 'source', 'cluster', 'prompt' columns

    Returns
    -------
    go.Figure
        Interactive grouped bar chart
    """
    pivot_table = dataset.pivot_table(
        index='source',
        columns='cluster',
        values='prompt',
        aggfunc='count',
        fill_value=0
    )

    fig = go.Figure()

    for cluster in pivot_table.columns:
        fig.add_trace(go.Bar(
            name=cluster,
            x=pivot_table.index,
            y=pivot_table[cluster],
            text=pivot_table[cluster],
            textposition='auto',
        ))

    fig.update_layout(
        title=dict(
            text='<b>Prompt Distribution by Source and Cluster</b>',
            x=0.5,
            font=dict(size=20, color='#ffffff')
        ),
        xaxis_title='<b>Data Source</b>',
        yaxis_title='<b>Number of Prompts</b>',
        barmode='group',
        template='plotly_dark',
        legend_title_text='Clusters',
        paper_bgcolor='rgba(15, 15, 35, 0.9)',
        plot_bgcolor='rgba(20, 20, 40, 0.8)',
        font=dict(color='#e0e0ff'),
        height=900,
        width=1600,
    )

    return fig
