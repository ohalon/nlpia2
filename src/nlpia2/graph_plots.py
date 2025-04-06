from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from nlpia2.nell import read_nell_tsv
import pandas as pd


def create_layout(G, layout='spring', **kwargs):
    if not layout:
        layout = 'spring'
    if isinstance(layout, str):
        layout = getattr(nx, f'{layout}_layout')
    if not callable(layout) and not isinstance(layout, dict):
        layout = nx.spring_layout
    layout = layout(G, **kwargs)  # draw graph
    return layout


def df_to_nx(df, columns='entity relation value prob'.split()):
    """ Convert a DataFrame for a knowledge graph (entity-relation-value triples) into a NetworkX Graph obj"""
    columns = columns or 'entity relation value prob'.split()
    triples = df[columns].T.values
    G = nx.DiGraph()
    # G.add_nodes_from(triples[0])
    # G.add_nodes_from(triples[2])
    # if len(columns) > 3:
    #     attrs = dict(zip(columns[3:], triples[3:]))
    G.add_edges_from(zip(triples[0], triples[2]), relation=triples[1])
    return G


def filter_triples_df(df, entities=None, relations=None, max_nodes=None, max_edges=None):
    return df


def plotly_edge_trace(G, layout=None, width=0.5, color='#888', hoverinfo='none', mode='lines', **scatter_kwargs):
    """ Create a plotly edge_trace for the edges in a NetworkX graph object """
    if layout is None:
        layout = 'spring'
    if isinstance(layout, str):
        layout = create_layout(layout)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = layout[edge[0]]
        x1, y1 = layout[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=width, color=color),
        hoverinfo=hoverinfo,
        mode=mode,
        **scatter_kwargs)


DEFAULT_NODE_MARKER = dict(
    showscale=True,
    # colorscale options
    #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
    #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
    #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
    colorscale='YlGnBu',
    reversescale=True,
    color=[],
    size=10,
    colorbar=dict(
        thickness=15,
        title='Node Connections',
        xanchor='left',
        titleside='right'
        ),
    line_width=2
    )


def plot_nx_graph(G, layout=None, show=True, node_color='b', width=2, with_labels=True, **kwargs):
    """ Plot a scatter plot of nodes in a NetworkX graph object """
    layout = create_layout(layout=layout)
    obj = nx.draw(G, layout, 
        node_color=node_color, width=width, with_labels=with_labels
    )  # draw edge attributes
    if show:
        plt.show()
    return obj


def plotly_node_trace(G, layout=None, marker=DEFAULT_NODE_MARKER, **layout_kwargs): 
    """ Create a plotly node_trace for the nodes in a NetworkX graph object """
    if layout is None:
        layout = 'spring'
    if isinstance(layout, str):
        layout = create_layout(layout, **layout_kwargs)
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = layout[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=marker)
    return node_trace


DEFAULT_ANNOTATION = dict(
            text="_Natural Language Processing in Action_ - [NLPiA2](https://pypi.org/project/NLPiA2)",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.005, y=-0.002 
            )


def plotly_graph(node_trace, edge_trace=None, layout='spring', path=None, show=None,
        title='Network (Graph)',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[DEFAULT_ANNOTATION],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)):
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Network (Graph)',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )
    """ Display a plotly network (graph) diagram defined by and edge_trace + node_trace """
    if show:
        if isinstance(show, (str, Path)) and not path:
            path = show
        else:
            fig.show()
            # !firefox knowledge.plotly.html
    if path:
        path = Path(path)
        if path.is_file():
            path.open('a').write(fig.to_html(html=False))
        else:
            path.open('a').write(fig.to_html())
    return fig.to_html()


def plot_graph(df_or_G=None, layout='spring', nrows=20000, filter_kwargs=None, **layout_kwargs):
    """ Plot a scatter plot of nodes and edges in a DataFrame(columns=graph object """
    if layout is None:
        layout = 'spring'
    if isinstance(layout, str):
        layout = create_layout(layout, **layout_kwargs)

    if df_or_G is None:
        df_or_G = read_nell_tsv(total=3_000_000, nrows=nrows)
    if isinstance(df_or_G, pd.DataFrame):
        if filter_kwargs:
            df_or_G = filter_triples_df(df_or_G, **filter_kwargs)
        G = df_to_nx(df_or_G)
    elif isinstance(df_or_G, (nx.Digraph, nx.Graph)):
        G = df_or_G
    edge_trace = plotly_edge_trace(G)
    node_trace = plotly_node_trace(G)
    html = plotly_graph(edge_trace=edge_trace, node_trace=node_trace, path=None)
    print(html)
    return html


def plot_subgraph(df, ent_match=r'(computer_science__)?artificial_intelligence', savefig=None):
    df = df[df['entity'].str.match(ent_match)]
    G = df_to_nx(df)
    layout = nx.spring_layout(G)
    nx.draw(G, pos=layout, node_color='#0343df', font_size=24,
        alpha=.5, width=2, with_labels=True)
    # nx.draw_edges(G, pos=layout, edge_color='g', font_size=20,
    #     alpha=.5, width=3, with_labels=True)
    # nx.draw_networkx_edge_labels(G, pos=layout, label_pos=0.75)
    if savefig:
        plt.savefig(savefig)
    plt.show(block=False)
    return df, G


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from nlpia2.nell import read_nell_tsv
    from nlpia2.graph_plots import df_to_nx
    import networkx as nx
    from nlpia2.graph_plots import plotly_node_trace, plotly_edge_trace, simplify_names
    df = read_nell_tsv(total=2_766_079, skiprows=None, nrows=250)
    labels = df.columns[:3]
    for c in labels:
        df[c] = df[c].str.split(':').apply(lambda x: x[-1])
    df[labels].head()
    df.shape
    G = df_to_nx(df)
    layout = nx.spring_layout(G)
    nx.draw(G, pos=layout, node_color='b', width=2, with_labels=True)
    plt.show(block=False)
    node_trace = plotly_node_trace(G=Gtiny, layout=layout)
    edge_trace = plotly_edge_trace(G=Gtiny, layout=layout)
    html = plot_graph(df)