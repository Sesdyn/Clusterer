"""
Plotting utilities for clustering results.

This module provides functions for visualizing clustering results including
dendrograms and cluster plots using Plotly for better performance.
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, TYPE_CHECKING
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import webbrowser
import threading
import time

if TYPE_CHECKING:
    from .clusterer import Cluster

def _plot_dendrogram(z: np.ndarray) -> None:
    """
    Plot hierarchical clustering dendrogram using matplotlib.
    
    This function visualizes the hierarchical clustering structure as a dendrogram,
    showing how clusters are formed at different distances.
    
    Parameters
    ----------
    z : np.ndarray
        Linkage matrix returned by scipy.cluster.hierarchy.linkage.
    """
    dendrogram(z, truncate_mode='lastp', show_leaf_counts=True, show_contracted=True)
    plt.show()


def plot_clusters(cluster_list: List["Cluster"], dist: str, mode: str = 'show', fname: str = 'results') -> None:
    """
    Plot cluster members on separate subplots using matplotlib.

    This function creates a grid of subplots where each subplot shows all time series
    belonging to a specific cluster. Each cluster is displayed in its own subplot
    with all member time series overlaid.

    Parameters
    ----------
    cluster_list : List[Cluster]
        List of Cluster objects containing clustered time series data.
    dist : str
        Distance metric name used for clustering. This will be displayed in the
        window title and can help identify which distance method was used.
    mode : str, default='show'
        Display mode for the plot:
        
        - 'show': Display the plot interactively using matplotlib.pyplot.show()
        - 'save': Save the plot to a PNG file without displaying it
        
    fname : str, default='results'
        Base filename for saving the plot (without extension). Only used when
        mode='save'. The file will be saved as '{fname}.png'.
    """
    main_fig = plt.figure(figsize=(14,10))
    main_fig.canvas.manager.set_window_title(dist + ' distance')
    no_plots = len(cluster_list)
    no_cols = 4
    no_rows = int(math.ceil(float(no_plots) / no_cols))
    i = 1
    
    for clust in cluster_list:
        sub_plot = main_fig.add_subplot(no_rows, no_cols, i)
        i = i + 1

        for each_ts in clust.list_of_members:
            t = np.arange(each_ts.data.shape[0])
            sub_plot.plot(t, each_ts.data, linewidth=2)
        
        plt.title('Cluster no: ' + str(clust.cluster_id), weight='bold')

    plt.tight_layout()
    if mode=='show':
        plt.show()
    elif mode=='save':
        plt.savefig('{0}.png'.format(fname))


def interactive_plot_clusters(cluster_list: List["Cluster"], dist: str, no_cols: int = 3, port: int = 8050) -> None:
    """
    Create an interactive plot of cluster members.

    When you click on a time series, information about that time series and its cluster will appear in the information panel.

    Press Ctrl+C to stop the server.

    Parameters
    ----------
    cluster_list : List[Cluster]
        List of Cluster objects containing clustered time series data.
    dist : str
        Distance metric name used for clustering.
    no_cols : int, default=3
        Number of columns in the plot.
    port : int, default=8050
        Port number for the Dash server.
    """
    
    app = dash.Dash(__name__)
    app.title = f"Interactive Clustering - {dist}"
    
    no_plots = len(cluster_list)
    no_cols = no_cols
    cluster_rows = int(math.ceil(float(no_plots) / no_cols))
    
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    
    cluster_fig = make_subplots(
        rows=cluster_rows, 
        cols=no_cols,
        subplot_titles=[f"Cluster {clust.cluster_id}" for clust in cluster_list],
        vertical_spacing=0.12,
        horizontal_spacing=0.04
    )
    
    cluster_data = {}
    time_series_data = {}
    
    for idx, clust in enumerate(cluster_list):
        row = (idx // no_cols) + 1
        col = (idx % no_cols) + 1

        cluster_data[clust.cluster_id] = {
            'cluster_id': clust.cluster_id,
            'number_of_members': clust.number_of_members,
            'indices_of_members': clust.indices_of_members.tolist(),
            'best_representative_member': clust.best_representative_member.label,
            'labels_of_members': [member.label for member in clust.list_of_members]
        }
        
        for j_idx, each_ts in enumerate(clust.list_of_members):
            t = np.arange(each_ts.data.shape[0])

            ts_id = f"cluster_{clust.cluster_id}_ts_{j_idx}"

            is_repr = each_ts.label == clust.best_representative_member.label

            time_series_data[ts_id] = {
                'name': each_ts.label,
                'cluster_id': clust.cluster_id,
                'length': len(each_ts.data),
                'mean': float(np.mean(each_ts.data)),
                'std': float(np.std(each_ts.data)),
                'min': float(np.min(each_ts.data)),
                'max': float(np.max(each_ts.data)),
                'is_representative': is_repr,
                'data': each_ts.data.tolist()
            }
            
            line_color = '#e74c3c' if is_repr else colors[j_idx % len(colors)]
            line_width = 2
            opacity = 0.8
            
            cluster_fig.add_trace(
                go.Scatter(
                    x=t,
                    y=each_ts.data,
                    mode='lines',
                    name=each_ts.label,
                    line=dict(width=line_width, color=line_color),
                    opacity=opacity,
                    customdata=[ts_id] * len(t),
                    hovertemplate=f"<b>{each_ts.label}</b><br>" +
                                "Time: %{x}<br>" +
                                "Value: %{y:.3f}<br>" +
                                f"Cluster: {clust.cluster_id}<br>" +
                                f"{'Representative' if is_repr else ''}<br>" +
                                "<extra></extra>",
                    showlegend=False
                ),
                row=row, col=col
            )
    
    cluster_fig.update_layout(
        height=280 * cluster_rows,
        showlegend=False,
        clickmode='event+select',
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font={'family': 'Arial, sans-serif'}
    )
    
    cluster_fig.update_annotations(font_size=18, font_color='#1a202c', font=dict(weight='bold'))
    
    info_panel_height = 160
    
    app.layout = html.Div([
        html.Div([
            html.H1([
                f"Interactive Clustering - {dist} distance"
            ], style={
                'textAlign': 'center', 
                'marginBottom': '4px',
                'color': '#1a202c',
                'fontFamily': 'Arial, sans-serif'
            })
        ]),
        
        html.Div([
            dcc.Graph(
                id='cluster-plot',
                figure=cluster_fig,
                style={'marginBottom': '2px'}
            )
        ], style={'marginTop': '0px'}),
        
        html.Div([
            html.Div([
                html.Div(id='info-panel', children=[])
            ], style={
                'backgroundColor': '#ffffff',
                'border': '1px solid #e2e8f0',
                'borderRadius': '16px',
                'padding': '18px',
                'margin': '4px auto',
                'width': 'calc(100% - 160px)',
                'boxShadow': '0 10px 25px rgba(0, 0, 0, 0.08)',
                'minHeight': f'{info_panel_height}px',
                'maxHeight': 'none',
                'overflow': 'visible',
                'boxSizing': 'border-box',
                'resize': 'both',
                'overflowY': 'visible'
            })
        ])
    ], style={
        'backgroundColor': '#f8f9fa',
        'minHeight': '100vh',
        'padding': '15px',
        'fontFamily': 'Arial, sans-serif'
    })
    
    @app.callback(
        Output('info-panel', 'children'),
        [Input('cluster-plot', 'clickData')]
    )
    def update_info_panel(clickData):
        if clickData is None:
            return [
                html.H2([
                    "Information Panel"
                ], style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#1a202c'}),
                html.Div([
                    html.Div([
                        html.P([
                            html.B("Representative members"), " are highlighted in ", 
                            html.Span("red", style={'color': '#e74c3c', 'fontWeight': 'bold'})
                        ], style={'textAlign': 'center', 'fontSize': '16px'}),
                        html.P("Click on any time series line to see detailed statistics", 
                               style={'textAlign': 'center', 'color': '#7f8c8d'})
                    ], style={
                        'padding': '36px',
                        'backgroundColor': '#ffffff',
                        'borderRadius': '12px',
                        'textAlign': 'center',
                        'border': '1px solid #e2e8f0'
                    })
                ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '100%'})
            ]
        
        try:
            point = clickData['points'][0]
            customdata = point.get('customdata')
            
            if customdata and customdata in time_series_data:
                ts_info = time_series_data[customdata]
                cluster_info = cluster_data[ts_info['cluster_id']]
                
                return [html.Div([
                    html.Div([
                        # Left column
                        html.Div([
                            html.Div([
                                html.Span("Time Series Information", style={'fontWeight': 'bold', 'color': '#34495e', 'fontSize': '16px'})
                            ], style={'marginBottom': '15px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Label:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '60px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['name']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Length:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '60px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['length']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Mean:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '60px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['mean']:.4f}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'})
                        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'overflow': 'hidden'}),
                        
                        # Middle column
                        html.Div([
                            html.Div([
                                html.Span("Std Dev:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '80px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['std']:.4f}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Max Value:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '80px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['max']:.4f}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Min Value:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '80px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['min']:.4f}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Representative:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '80px', 'display': 'inline-block'}),
                                html.Span(" Yes" if ts_info["is_representative"] else " No", 
                                         style={'marginLeft': '10px', 'color': '#34495e' if ts_info["is_representative"] else '#34495e', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'})
                        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%', 'overflow': 'hidden'}),
                        
                        # Gray vertical line
                        html.Div(style={'width': '1px', 'backgroundColor': '#cccccc', 'height': '120px', 'display': 'inline-block', 'marginLeft': '2%', 'marginRight': '2%', 'verticalAlign': 'top'}),
                        
                        # Right column
                        html.Div([
                            html.Div([
                                html.Span("Cluster Information", style={'fontWeight': 'bold', 'color': '#34495e', 'fontSize': '16px'})
                            ], style={'marginBottom': '15px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Cluster ID:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '100px', 'display': 'inline-block'}),
                                html.Span(f" {ts_info['cluster_id']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Cluster Size:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '100px', 'display': 'inline-block'}),
                                html.Span(f" {cluster_info['number_of_members']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'}),
                            html.Div([
                                html.Span("Representative Member:", style={'fontWeight': 'bold', 'color': '#34495e', 'minWidth': '100px', 'display': 'inline-block'}),
                                html.Span(f" {cluster_info['best_representative_member']}", style={'marginLeft': '10px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '10px', 'overflow': 'hidden'})
                        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'overflow': 'hidden'})
                    ], style={'wordWrap': 'break-word'})
                ])]
        except (KeyError, IndexError, TypeError):
            return [
                html.Div([
                    html.P("Error: Unable to display information for the selected time series.", 
                           style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '16px'})
                ], style={'padding': '20px'})
            ]

    app.cluster_data = cluster_data
    app.time_series_data = time_series_data
    
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        webbrowser.open(f'http://127.0.0.1:{port}')

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        app.run(debug=False, port=port, host='127.0.0.1', 
                dev_tools_silence_routes_logging=True, 
                use_reloader=False)
    except Exception as e:
        print(f"Error starting server: {e}")




def multiple_tabs_interactive_plot_clusters(cluster_list: List["Cluster"], dist: str, port: int = 8051) -> None:
    """
    Create an interactive plot of cluster members, each cluster is displayed in its own tab.

    When you click on a time series, information about that time series and its cluster will appear in the information panel.

    Additionally includes a "Representatives" tab showing all cluster representatives in one plot.

    Press Ctrl+C to stop the server.

    Parameters
    ----------
    cluster_list : List[Cluster]
        List of Cluster objects containing clustered time series data.
    dist : str
        Distance metric name used for clustering.
    port : int, default=8051
        Port number for the Dash server.
    """
    
    app = dash.Dash(__name__)
    app.title = f"Interactive Clustering Tabs - {dist}"
    
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    
    cluster_data = {}
    time_series_data = {}
    representative_data = {}
    
    for clust in cluster_list:
        cluster_data[clust.cluster_id] = {
            'cluster_id': clust.cluster_id,
            'number_of_members': clust.number_of_members,
            'indices_of_members': clust.indices_of_members.tolist(),
            'best_representative_member': clust.best_representative_member.label,
            'labels_of_members': [member.label for member in clust.list_of_members]
        }
        
        representative_ts = clust.best_representative_member
        repr_id = f"repr_cluster_{clust.cluster_id}"
        representative_data[repr_id] = {
            'name': representative_ts.label,
            'cluster_id': clust.cluster_id,
            'data': representative_ts.data,
            'length': len(representative_ts.data),
            'mean': float(np.mean(representative_ts.data)),
            'std': float(np.std(representative_ts.data)),
            'min': float(np.min(representative_ts.data)),
            'max': float(np.max(representative_ts.data)),
            'feature_vector': representative_ts.feature_vector
        }
        
        for j_idx, each_ts in enumerate(clust.list_of_members):
            ts_id = f"cluster_{clust.cluster_id}_ts_{j_idx}"
            time_series_data[ts_id] = {
                'name': each_ts.label,
                'cluster_id': clust.cluster_id,
                'length': len(each_ts.data),
                'mean': float(np.mean(each_ts.data)),
                'std': float(np.std(each_ts.data)),
                'min': float(np.min(each_ts.data)),
                'max': float(np.max(each_ts.data)),
                'is_representative': each_ts.label == clust.best_representative_member.label,
                'data': each_ts.data.tolist(),
                'feature_vector': each_ts.feature_vector
            }

    tabs = []
    
    # Individual cluster tabs
    for clust in cluster_list:
        tab_content = _create_cluster_tab_content(clust, colors, time_series_data, cluster_data)
        tabs.append(dcc.Tab(
            label=f'Cluster {clust.cluster_id}',
            value=f'cluster-{clust.cluster_id}',
            children=tab_content,
            style={
                'padding': '8px 16px', 
                'fontWeight': 'bold',
                'fontSize': '14px',
                'border': 'none',
                'borderRadius': '8px 8px 0 0',
                'margin': '0 2px'
            },
            selected_style={
                'padding': '8px 16px', 
                'fontWeight': 'bold', 
                'backgroundColor': '#119DFF', 
                'color': 'white',
                'fontSize': '14px',
                'border': 'none',
                'borderRadius': '8px 8px 0 0',
                'margin': '0 2px'
            }
        ))
    
    # Representatives tab
    repr_tab_content = _create_representatives_tab_content(cluster_list, colors, representative_data, cluster_data)
    tabs.append(dcc.Tab(
        label='Representatives',
        value='representatives',
        children=repr_tab_content,
        style={
            'padding': '8px 16px', 
            'fontWeight': 'bold',
            'fontSize': '14px',
            'border': 'none',
            'borderRadius': '8px 8px 0 0',
            'margin': '0 2px'
        },
        selected_style={
            'padding': '8px 16px', 
            'fontWeight': 'bold', 
            'backgroundColor': '#119DFF', 
            'color': 'white',
            'fontSize': '14px',
            'border': 'none',
            'borderRadius': '8px 8px 0 0',
            'margin': '0 2px'
        }
    ))

    # Main layout
    app.layout = html.Div([
        html.Div([
            html.H1(f"Interactive Clustering Tabs - {dist} distance", 
                   style={
                       'textAlign': 'center', 
                       'marginBottom': '25px', 
                       'color': '#1a202c',
                       'fontFamily': 'Arial, sans-serif',
                       'fontSize': '28px',
                       'fontWeight': 'bold'
                   })
        ]),
        
        html.Div([
            dcc.Tabs(
                id="main-tabs",
                value=f'cluster-{cluster_list[0].cluster_id}',
                children=tabs,
                style={
                    'height': '50px',
                    'fontFamily': 'Arial, sans-serif'
                },
                colors={
                    'border': 'white', 
                    'primary': '#119DFF', 
                    'background': '#f8f9fa'
                }
            )
        ], style={
            'backgroundColor': '#ffffff',
            'borderRadius': '12px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'overflow': 'hidden',
            'marginBottom': '20px'
        })
    ], style={
        'backgroundColor': '#f8f9fa', 
        'minHeight': '100vh', 
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif'
    })

    # Callbacks for each cluster tab
    for clust in cluster_list:
        _create_cluster_callbacks(app, clust.cluster_id, time_series_data, cluster_data)
    
    # Callback for representatives tab
    _create_representatives_callback(app, representative_data, cluster_data)
    
    # Auto-open browser
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f'http://127.0.0.1:{port}')

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        app.run(debug=False, port=port, host='127.0.0.1', 
                dev_tools_silence_routes_logging=True, 
                use_reloader=False)
    except Exception as e:
        print(f"Error starting server: {e}")


def _create_cluster_tab_content(clust, colors, time_series_data, cluster_data):
    """Create content for individual cluster tab."""
    
    # First row: Cluster info panel (left) and representative plot (right)
    repr_name = clust.best_representative_member.label
    repr_data = clust.best_representative_member.data
    repr_fig = go.Figure()
    t = np.arange(repr_data.shape[0])
    
    repr_fig.add_trace(go.Scatter(
        x=t, y=repr_data,
        mode='lines',
        name=repr_name,
        line=dict(width=3, color='#e74c3c'),
        hovertemplate=f"<b>{repr_name}</b><br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
    ))
    
    repr_fig.update_layout(
        title=f"Representative Member",
        height=280,
        showlegend=False,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        autosize=True,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Second row: All time series
    all_series_fig = go.Figure()
    
    for j_idx, each_ts in enumerate(clust.list_of_members):
        t = np.arange(each_ts.data.shape[0])
        is_repr = each_ts.label == clust.best_representative_member.label
        line_color = '#e74c3c' if is_repr else colors[j_idx % len(colors)]
        line_width = 3.5 if is_repr else 2.5
        opacity = 1.0 if is_repr else 0.8
        ts_id = f"cluster_{clust.cluster_id}_ts_{j_idx}"
        
        all_series_fig.add_trace(go.Scatter(
            x=t, y=each_ts.data,
            mode='lines',
            name=f'{each_ts.label}',
            line=dict(width=line_width, color=line_color),
            opacity=opacity,
            customdata=[ts_id] * len(t),
            hovertemplate=f"<b>{each_ts.label}</b><br>Time: %{{x}}<br>Value: %{{y:.3f}}<br>{'Representative' if is_repr else ''}<extra></extra>"
        ))
    
    all_series_fig.update_layout(
        title=f"All Time Series in Cluster {clust.cluster_id}",
        height=400,
        showlegend=False,
        clickmode='event+select',
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa'
    )
    
    # Create cluster info panel content
    cluster_info_content = html.Div([
        html.H3("Cluster Information", style={
            'color': '#1a202c', 
            'marginBottom': '15px',
            'fontSize': '18px',
            'fontWeight': 'bold'
        }),
        html.P([html.Strong("Cluster ID: "), str(clust.cluster_id)], style={
            'marginBottom': '10px',
            'fontSize': '14px',
            'lineHeight': '1.5'
        }),
        html.P([html.Strong("Number of Members: "), str(clust.number_of_members)], style={
            'marginBottom': '10px',
            'fontSize': '14px',
            'lineHeight': '1.5'
        }),
        html.P([html.Strong("Representative Member: "), str(clust.best_representative_member.label)], style={
            'marginBottom': '8px',
            'fontSize': '14px',
            'lineHeight': '1.5',
            'wordBreak': 'break-word'
        })
    ], style={
        'backgroundColor': '#ffffff',
        'border': '1px solid #e2e8f0',
        'borderRadius': '12px',
        'padding': '20px',
        'height': '260px',
        'overflow': 'auto',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'position': 'relative',
        'zIndex': '1'
    })
    
    return html.Div([
        # First row: Info panel (left) and representative plot (right)
        html.Div([
            html.Div([cluster_info_content], style={
                'width': '30%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'paddingRight': '15px',
                'boxSizing': 'border-box'
            }),
            html.Div([
                dcc.Graph(
                    figure=repr_fig,
                    style={'height': '280px'},
                    config={'displayModeBar': False, 'responsive': True}
                )
            ], style={
                'width': '70%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'boxSizing': 'border-box',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '12px',
                'padding': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={
            'marginBottom': '20px',
            'display': 'flex',
            'alignItems': 'flex-start',
            'gap': '15px'
        }),
        
        # Second row: All time series
        html.Div([
            dcc.Graph(
                id=f'all-series-plot-{clust.cluster_id}',
                figure=all_series_fig,
                style={'width': '100%', 'height': '400px'},
                config={'displayModeBar': False, 'responsive': True}
            )
        ], style={
            'marginBottom': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '12px',
            'padding': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        
        # Third row: Detailed statistics panel
        html.Div([
            html.Div(id=f'stats-panel-{clust.cluster_id}', children=_create_default_stats_panel())
        ], style={
            'backgroundColor': '#ffffff',
            'border': '1px solid #e2e8f0',
            'borderRadius': '12px',
            'padding': '20px',
            'minHeight': '180px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'position': 'relative',
            'zIndex': '10',
            'marginTop': '20px'
        })
    ], style={
        'position': 'relative',
        'zIndex': '1',
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'minHeight': 'auto'
    })


def _create_representatives_tab_content(cluster_list, colors, representative_data, cluster_data):
    """Create content for representatives tab."""
    
    repr_fig = go.Figure()
    
    for idx, clust in enumerate(cluster_list):
        repr_name = clust.best_representative_member.label
        repr_data = clust.best_representative_member.data

        t = np.arange(repr_data.shape[0])
        
        repr_id = f"repr_cluster_{clust.cluster_id}"
        
        repr_fig.add_trace(go.Scatter(
            x=t, y=repr_data,
            mode='lines',
            name=f'Cluster {clust.cluster_id} Rep',
            line=dict(width=3, color=colors[idx % len(colors)]),
            customdata=[repr_id] * len(t),
            hovertemplate=f"<b>Cluster {clust.cluster_id} Representative</b><br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
        ))
    
    repr_fig.update_layout(
        title="Cluster Representatives",
        height=500,
        showlegend=True,
        clickmode='event+select',
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa'
    )
    
    return html.Div([
        html.Div([
            dcc.Graph(
                id='representatives-plot',
                figure=repr_fig,
                style={'height': '500px'},
                config={'displayModeBar': False, 'responsive': True}
            )
        ], style={
            'backgroundColor': '#f8f9fa',
            'borderRadius': '12px',
            'padding': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px'
        }),
        html.Div([
            html.Div(id='representatives-info-panel', children=_create_default_representatives_panel())
        ], style={
            'backgroundColor': '#ffffff',
            'border': '1px solid #e2e8f0',
            'borderRadius': '12px',
            'padding': '20px',
            'minHeight': '200px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'position': 'relative',
            'zIndex': '10',
            'marginTop': '20px'
        })
    ], style={
        'position': 'relative',
        'zIndex': '1',
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'minHeight': 'auto'
    })


def _create_default_stats_panel():
    """Create default statistics panel."""
    return [
        html.H4("Time Series Statistics", style={
            'textAlign': 'center', 
            'marginBottom': '20px', 
            'color': '#1a202c',
            'fontSize': '18px',
            'fontWeight': 'bold'
        }),
        html.P("Click on any time series line above to see detailed statistics", 
               style={
                   'textAlign': 'center', 
                   'color': '#7f8c8d',
                   'fontSize': '14px',
                   'lineHeight': '1.5'
               })
    ]


def _create_default_representatives_panel():
    """Create default representatives information panel."""
    return [
        html.H4("Cluster Information", style={
            'textAlign': 'center', 
            'marginBottom': '20px', 
            'color': '#1a202c',
            'fontSize': '18px',
            'fontWeight': 'bold'
        }),
        html.P("Click on any representative line above to see cluster information", 
               style={
                   'textAlign': 'center', 
                   'color': '#7f8c8d',
                   'fontSize': '14px',
                   'lineHeight': '1.5'
               })
    ]


def _create_cluster_callbacks(app, cluster_id, time_series_data, cluster_data):
    """Create callbacks for individual cluster tabs."""
    
    @app.callback(
        Output(f'stats-panel-{cluster_id}', 'children'),
        [Input(f'all-series-plot-{cluster_id}', 'clickData')]
    )
    def update_stats_panel(clickData):
        if clickData is None:
            return _create_default_stats_panel()
        
        try:
            point = clickData['points'][0]
            customdata = point.get('customdata')
            
            if customdata and customdata in time_series_data:
                ts_info = time_series_data[customdata]
                
                # Print information to terminal
                print(f"\nTime Series Clicked:")
                print(f"Label: {ts_info['name']}")
                print(f"Data: {ts_info['data']}")
                print(f"Feature vector: {ts_info['feature_vector']}")
                print(f"Cluster id: {ts_info['cluster_id']}")
                print("-" * 80)
                
                return [
                    html.Div([
                        # Left column
                        html.Div([
                            html.H4("Time Series Information", style={
                                'color': '#34495e', 
                                'marginBottom': '15px',
                                'fontSize': '16px',
                                'fontWeight': 'bold'
                            }),
                            html.P([html.Strong("Label: "), ts_info['name']], style={
                                'marginBottom': '10px',
                                'fontSize': '14px',
                                'lineHeight': '1.4'
                            }),
                            html.P([html.Strong("Length: "), str(ts_info['length'])], style={
                                'marginBottom': '10px',
                                'fontSize': '14px',
                                'lineHeight': '1.4'
                            }),
                            html.P([html.Strong("Mean: "), f"{ts_info['mean']:.4f}"], style={
                                'marginBottom': '10px',
                                'fontSize': '14px',
                                'lineHeight': '1.4'
                            }),
                            html.P([html.Strong("Representative: "), "Yes" if ts_info['is_representative'] else "No"], style={
                                'marginBottom': '8px',
                                'fontSize': '14px',
                                'lineHeight': '1.4'
                            })
                        ], style={
                            'width': '35%', 
                            'flexShrink': '0',
                            'paddingRight': '15px'
                        }),
                        
                        # Right column
                        html.Div([
                            html.H4("Statistical Details", style={
                                'color': '#34495e', 
                                'marginBottom': '15px',
                                'fontSize': '16px',
                                'fontWeight': 'bold'
                            }),
                            html.P([html.Strong("Std Dev: "), f"{ts_info['std']:.4f}"], style={
                                'marginBottom': '10px',
                                'fontSize': '14px',
                                'lineHeight': '1.4'
                            }),
                            html.P([html.Strong("Max Value: "), f"{ts_info['max']:.4f}"], style={
                                'marginBottom': '10px',
                                'fontSize': '14px',
                                'lineHeight': '1.4'
                            }),
                            html.P([html.Strong("Min Value: "), f"{ts_info['min']:.4f}"], style={
                                'marginBottom': '10px',
                                'fontSize': '14px',
                                'lineHeight': '1.4'
                            }),
                            html.P([html.Strong("Range: "), f"{ts_info['max'] - ts_info['min']:.4f}"], style={
                                'marginBottom': '8px',
                                'fontSize': '14px',
                                'lineHeight': '1.4'
                            })
                        ], style={
                            'width': '62%', 
                            'flexShrink': '0',
                            'marginLeft': '3%',
                            'paddingLeft': '50px',
                            'borderLeft': '1px solid #e2e8f0'
                        })
                    ], style={
                        'position': 'relative',
                        'zIndex': '1',
                        'display': 'flex',
                        'flexDirection': 'row',
                        'alignItems': 'flex-start',
                        'width': '100%'
                    })
                ]
        except (KeyError, IndexError, TypeError):
            return [
                html.P("Error: Unable to display statistics for the selected time series.", 
                       style={'textAlign': 'center', 'color': '#e74c3c'})
            ]


def _create_representatives_callback(app, representative_data, cluster_data):
    """Create callback for representatives tab."""
    
    @app.callback(
        Output('representatives-info-panel', 'children'),
        [Input('representatives-plot', 'clickData')]
    )
    def update_representatives_panel(clickData):
        if clickData is None:
            return _create_default_representatives_panel()
        
        try:
            point = clickData['points'][0]
            customdata = point.get('customdata')
            
            if customdata and customdata in representative_data:
                repr_info = representative_data[customdata]
                cluster_info = cluster_data[repr_info['cluster_id']]
                
                # Print information to terminal
                print(f"\nRepresentative Time Series Clicked:")
                print(f"Label: {repr_info['name']}")
                print(f"Data: {repr_info['data'].tolist()}")
                print(f"Feature vector: {repr_info['feature_vector']}")
                print(f"Cluster id: {repr_info['cluster_id']}")
                print("-" * 80)
                
                return [
                    html.Div([
                        # Cluster details
                        html.Div([
                            html.Div([
                                html.H4("Cluster Details", style={
                                    'color': '#34495e', 
                                    'marginBottom': '15px',
                                    'fontSize': '16px',
                                    'fontWeight': 'bold'
                                }),
                                html.P([html.Strong("Cluster ID: "), str(cluster_info['cluster_id'])], style={
                                    'marginBottom': '10px',
                                    'fontSize': '14px',
                                    'lineHeight': '1.4'
                                }),
                                html.P([html.Strong("Total Members: "), str(cluster_info['number_of_members'])], style={
                                    'marginBottom': '10px',
                                    'fontSize': '14px',
                                    'lineHeight': '1.4'
                                }),
                                html.P([html.Strong("Representative Label: "), str(cluster_info['best_representative_member'])], style={
                                    'marginBottom': '8px',
                                    'fontSize': '14px',
                                    'lineHeight': '1.4',
                                    'wordBreak': 'break-word'
                                })
                            ], style={
                                'width': '48%', 
                                'flexShrink': '0',
                                'paddingRight': '15px'
                            }),
                            
                            html.Div([
                                html.H4("Statistical Details", style={
                                    'color': '#34495e', 
                                    'marginBottom': '15px',
                                    'fontSize': '16px',
                                    'fontWeight': 'bold'
                                }),
                                html.P([html.Strong("Length: "), str(repr_info['length'])], style={
                                    'marginBottom': '10px',
                                    'fontSize': '14px',
                                    'lineHeight': '1.4'
                                }),
                                html.P([html.Strong("Mean: "), f"{repr_info['mean']:.4f}"], style={
                                    'marginBottom': '10px',
                                    'fontSize': '14px',
                                    'lineHeight': '1.4'
                                }),
                                html.P([html.Strong("Std Dev: "), f"{repr_info['std']:.4f}"], style={
                                    'marginBottom': '10px',
                                    'fontSize': '14px',
                                    'lineHeight': '1.4'
                                })
                            ], style={
                                'width': '48%', 
                                'flexShrink': '0',
                                'marginLeft': '4%',
                                'paddingLeft': '50px',
                                'borderLeft': '1px solid #e2e8f0'
                            })
                        ], style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'alignItems': 'flex-start',
                            'width': '100%'
                        })
                    ], style={
                        'position': 'relative',
                        'zIndex': '1'
                    })
                ]
        except (KeyError, IndexError, TypeError):
            return [
                html.P("Error: Unable to display information for the selected representative.", 
                       style={'textAlign': 'center', 'color': '#e74c3c'})
            ]