import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_rl(pc_r, seg_r, pc_l, seg_l, show=True):
    """ Plot receptor and ligand as point clouds with Plotly
    :param pc_r: receptor point cloud [n_verts, 3]
    :param seg_r: segmentation of receptor point cloud [n_verts, 1]
    :param pc_l: ligand point cloud [n_verts, 3]
    :param seg_l: segmentation of ligand point cloud [n_verts, 1]
    :param show: show plot
    :return: None
    """

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("receipt", "ligand"),
        specs=[[{'type': 'scene'}, {'type': 'scene'}]])

    pc_r1 = pc_r[seg_r == 1, :]
    pc_r2 = pc_r[seg_r==2, :]
    fig.add_traces([go.Scatter3d(x=pc_r1[:, 0], y=pc_r1[:, 1], z=pc_r1[:, 2], name="receipt_1", mode='markers',
                               marker=dict(
                                   size=5,
                                   color=np.arange(pc_r1.shape[0]),  # set color to an array/list of desired values
                                   colorscale='Plasma'
                               )),
                    go.Scatter3d(x=pc_r2[:, 0], y=pc_r2[:, 1], z=pc_r2[:, 2], name="receipt_2", mode='markers',
                                 marker=dict(
                                     size=5,
                                     color=np.arange(pc_r2.shape[0]),  # set color to an array/list of desired values
                                     colorscale='Plasma'
                                 ))
                    ], rows=1, cols=1)
    pc_l1 = pc_l[seg_l==1, :]
    pc_l2 = pc_l[seg_l==2, :]
    fig.add_traces([go.Scatter3d(x=pc_l1[:, 0], y=pc_l1[:, 1], z=pc_l1[:, 2], name="ligand_1", mode='markers',
                               marker=dict(
                                   size=5,
                                   color=np.arange(pc_l1.shape[0]),  # set color to an array/list of desired values
                                   colorscale='Plasma'
                               )),
                    go.Scatter3d(x=pc_l2[:, 0], y=pc_l2[:, 1], z=pc_l2[:, 2], name="ligand_2", mode='markers',
                                 marker=dict(
                                     size=5,
                                     color=np.arange(pc_l2.shape[0]),  # set color to an array/list of desired values
                                     colorscale='Plasma'
                                 ))
                    ], rows=1, cols=2)

    if show:
        fig.show()
    return fig




def plot_rl_colors(pc_r, col_r, pc_l, col_l, show=True):
    """
    Plot receptor and ligand as point clouds with Plotly

    :param pc_r:    receptor point cloud
    :param col_r:   receptor colors
    :param pc_l:    ligand point cloud
    :param col_l:   ligand colors
    :param show:    show plot
    :return:        plotly figure
    """

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("receipt", "ligand"),
        specs=[[{'type': 'scene'}, {'type': 'scene'}]])

    fig.add_traces([go.Scatter3d(x=pc_r[:, 0], y=pc_r[:, 1], z=pc_r[:, 2], name="receipt_1", mode='markers',
                                 marker=dict(
                                     size=5,
                                     color=col_r,  # set color to an array/list of desired values
                                     colorscale='Plasma'
                                 )), ], rows=1, cols=1)

    fig.add_traces([go.Scatter3d(x=pc_l[:, 0], y=pc_l[:, 1], z=pc_l[:, 2], name="ligand_1", mode='markers',
                                 marker=dict(
                                     size=5,
                                     color=col_l,  # set color to an array/list of desired values
                                     colorscale='Plasma'
                                 )), ], rows=1, cols=2)

    if show:
        fig.show()
    return fig


def plot_mesh(v, f, params, show=True):
    """  Plot a mesh with Plotly
    https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Mesh3d.html
    :param v:  vertices
    :param f:   faces
    :param params:  parameters for the plotly mesh
    :param show:    show plot
    :return:    plotly figure
    """
    fig = go.Figure(data=[go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                    contour_width=2, contour_color="#101010", contour_show=True,
                                    **params)])

    fig.update_traces(lighting=dict(ambient=0.8, diffuse=0.5,
                                    specular=0.5, roughness=0.5,
                                    fresnel=0.5))
    if show:
        fig.show()
    return fig


def plot_pointcloud(coords, color=None, show=True):
    """ Plot a point cloud with Plotly

    :param coords: numpy ndarray with 3d coordinates, size: [n,3]
    :param color: array of values to plot as color. When None, is derived from the coordinates. Size: [n]. Default: None
    :return: plotly figure

    """
    if color is None:
        color = np.arange(coords.shape[0])
    fig = go.Figure(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode='markers',
                                 marker=dict(
                                     size=5,
                                     color=color,  # set color to an array/list of desired values
                                     colorscale='Plasma'
                                 )))
    if show:
        fig.show()
    return fig


# def plot_meshes(v, f, params, show=True):
#     # fig = plot_mesh(v,f, {'opacity':1,'contour' : go.mesh3d.Contour()})
#     # fig = plot_meshes((Vm,Vn),(Fm,Fn),({'name':"Target",'colorscale':"Emrld",'intensity':Vm[:,2], 'opacity': 0.9},
#     #                                    {'name':"N",'colorscale':"Peach",'intensity':Vn[:,2]}), show = False)
#     fig = go.Figure()
#     for i_v in range(len(v)):
#         fig.add_mesh3d(x=v[i_v][:, 0], y=v[i_v][:, 1], z=v[i_v][:, 2], i=f[i_v][:, 0], j=f[i_v][:, 1], k=f[i_v][:, 2],
#                        contour_width=2, contour_color="#101010", contour_show=True,
#                        **params[i_v])
#
#     fig.update_traces(lighting=dict(ambient=0.8, diffuse=0.5,
#                                     specular=0.5, roughness=0.5,
#                                     fresnel=0.5))
#     if show:
#         fig.show()
#     return fig


def plot_colors(pc_r, col_r, pc_l, col_l, subplots_titles, show=True, colorscale='Plasma'):
    """  Plot receptor and ligand as point clouds with Plotly

    :param pc_r:    receptor point cloud
    :param col_r:   receptor colors
    :param pc_l:    ligand point cloud
    :param col_l:   ligand colors
    :param subplots_titles: titles for the subplots
    :param show:    show plot
    :param colorscale:  colorscale for the plot
    :return:        plotly figure
    """

    cmax = np.max(np.concatenate((col_r, col_l), axis=0))
    cmin = np.min(np.concatenate((col_r, col_l), axis=0))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]])

    fig.add_traces([go.Scatter3d(x=pc_r[:, 0], y=pc_r[:, 1], z=pc_r[:, 2], mode='markers',
                                 marker=dict(
                                     size=5, color=col_r,  # cmax=cmax, cmin=cmin,
                                     colorscale=colorscale,  # colorbar=dict(thickness=20)
                                 )), ], rows=1, cols=1)

    fig.add_traces([go.Scatter3d(x=pc_l[:, 0], y=pc_l[:, 1], z=pc_l[:, 2], mode='markers',
                                 marker=dict(
                                     size=5, color=col_l, cmax=cmax, cmin=cmin,
                                     colorscale=colorscale, colorbar=dict(thickness=20)
                                 )), ], rows=1, cols=2)

    if show:
        fig.show()
    return fig


def plot_pointclouds(v, cols, subplots_titles, colorscale='Plasma', size=5, show=True):
    """ Plot point clouds with Plotly

        :param v:   list of point clouds
        :param cols:    list of colors for each point cloud
        :param subplots_titles: titles for the subplots
        :param colorscale:  colorscale for the plot
        :param size:    size of the points
        :param show:    show plot
        :return:        plotly figure
        """
    fig = make_subplots(
        rows=1, cols=len(v),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'} for i in range(len(v))]])



    fig.layout.coloraxis.colorscale = colorscale
    for i in range(len(v)):
        fig.layout[f"scene{i + 1}"].update(dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
        ))
    for i_v in range(len(v)):
        fig.add_traces([go.Scatter3d(x=v[i_v][:, 0], y=v[i_v][:, 1], z=v[i_v][:, 2],
                                     mode='markers',
                                     marker=dict(
                                         size=size, color=cols[i_v], coloraxis="coloraxis",
                                         showscale=False,
                                         # cmax=cmax, cmin=cmin, colorscale=colorscale,
                                     )), ], rows=1, cols=i_v + 1)

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None],
                                  mode='markers', marker=dict(
            # colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, coloraxis="coloraxis",
            colorbar=dict(thickness=10, outlinewidth=0)
        ), hoverinfo='none')
    fig['layout']['showlegend'] = False
    fig.add_traces([colorbar_trace, ], rows=1, cols=i_v + 1)

    if show:
        fig.show()
    return fig

def plot_abag_pointclouds(v, cols, subplots_titles,  size=5, show=True, x_eye = -1.25, y_eye = 2, z_eye = 0.5):
    """ Plot point clouds with Plotly

    :param v:   list of point clouds
    :param cols:    list of colors for each point cloud
    :param subplots_titles: titles for the subplots
    :param colorscale:  colorscale for the plot
    :param size:    size of the points
    :param show:    show plot
    :return:        plotly figure
    """

    fig = make_subplots(
        rows=1, cols=len(v),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'} for i in range(len(v))]])

    fig.update_layout(coloraxis={'colorscale': 'Blues', 'cmax': 1, 'cmin': -0.4})  # ab
    fig.update_layout(coloraxis2={'colorscale': 'Reds', 'cmax': 1, 'cmin':  -0.4})  # ag

    for i in range(len(v)):
        fig.layout[f"scene{i + 1}"].update(dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
            camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        ))
    for i_v in range(len(v)):
        traces = list()
        mesh = v[i_v].get("ab", None)
        if mesh is not None:
            color = cols[i_v].get("ab", None)
            # ab
            traces.append(go.Scatter3d(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2], mode='markers',
                                     marker=dict(
                                         size=size, color=color, coloraxis="coloraxis", showscale=False,
                                     ),line={"color":'rgb(125,125,125)',"width": 0.5}))
        mesh = v[i_v].get("ag", None)
        if mesh is not None:
            color = cols[i_v].get("ag", None)
            # ab
            traces.append(go.Scatter3d(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2],mode='markers',
                                     marker=dict(
                                         size=size, color=color, coloraxis="coloraxis2", showscale=False,
                                     ),line={"color":'rgb(125,125,125)',"width": 0.5}))
        fig.add_traces(traces, rows=1, cols=i_v + 1)

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None],
                                  mode='markers', marker=dict(
            # colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, coloraxis="coloraxis",
            colorbar=dict(thickness=10, outlinewidth=0)
        ), hoverinfo='none')
    fig['layout']['showlegend'] = False
    fig.add_traces([colorbar_trace, ], rows=1, cols=i_v + 1)

    if show:
        fig.show()
    return fig

def plot_3dgraphs(v, edges, cols, subplots_titles, colorscale='Plasma', size=5, show=True):
    """ Plot 3D graphs with Plotly

    :param v:   list of graphs
    :param edges:   list of edges
    :param cols:    list of colors for each graph
    :param subplots_titles: titles for the subplots
    :param colorscale:  colorscale for the plot
    :param size:    size of the points
    :param show:    show plot
    :return:        plotly figure
    """

    fig = make_subplots(
        rows=1, cols=len(v),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'} for i in range(len(v))]])

    fig.layout.coloraxis.colorscale = colorscale
    for i in range(len(v)):
        fig.layout[f"scene{i + 1}"].update(dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
        ))
    for i_v in range(len(v)):
        fig.add_traces([go.Scatter3d(x=v[i_v][:, 0], y=v[i_v][:, 1], z=v[i_v][:, 2],
                                     mode='markers',
                                     marker=dict(
                                         size=size, color=cols[i_v], coloraxis="coloraxis",
                                         showscale=False,
                                         # cmax=cmax, cmin=cmin, colorscale=colorscale,
                                     ))]  +
                       [go.Scatter3d(x=[v[i_v][edges[i_v][0,i_l], 0], v[i_v][edges[i_v][1,i_l], 0]],
                                     y=[v[i_v][edges[i_v][0,i_l], 1], v[i_v][edges[i_v][1,i_l], 1]],
                                     z=[v[i_v][edges[i_v][0,i_l], 2], v[i_v][edges[i_v][1,i_l], 2]],
                                     showlegend=False, mode='lines',
                                     line=dict(
                                         color='rgb(125,125,125)', dash='dash',
                                         width=1, showscale=False
                                     )) for i_l in range(edges[i_v].shape[1])
                        ], rows=1, cols=i_v + 1)

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None],
                                  mode='markers', marker=dict(
            # colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, coloraxis="coloraxis",
            colorbar=dict(thickness=10, outlinewidth=0)
        ), hoverinfo='none')
    fig['layout']['showlegend'] = False
    fig.add_traces([colorbar_trace, ], rows=1, cols=i_v + 1)

    if show:
        fig.show()
    return fig


def plot_abag_3dgraphs(v, edges, cols, subplots_titles, size=5, show=True, x_eye = -1.25, y_eye = 2, z_eye = 0.5,
                       cmax=1, cmin=-0.4):
    """     Plot 3D graphs with Plotly

    :param v:   list of graphs
    :param edges:   list of edges
    :param cols:    list of colors for each graph
    :param subplots_titles:     titles for the subplots
    :param size:    size of the points
    :param show:    show plot
    :param x_eye:   x position of the camera
    :param y_eye:   y position of the camera
    :param z_eye:   z position of the camera
    :param cmax:    max value for the colorbar
    :param cmin:    min value for the colorbar
    :return:
    """

    fig = make_subplots(
        rows=1, cols=len(v),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'} for i in range(len(v))]])

    fig.update_layout(coloraxis={'colorscale': 'Blues', 'cmax':cmax, 'cmin': cmin}) # ab
    fig.update_layout(coloraxis2={'colorscale': 'Reds', 'cmax':cmax, 'cmin': cmin}) # ag

    for i in range(len(v)):
        fig.layout[f"scene{i + 1}"].update(dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
            camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        ))
    for i_v in range(len(v)):
        traces = list()
        nodes = v[i_v].get("ab", None)
        if nodes is not None:
            color = cols[i_v].get("ab", None)
            edge = edges[i_v].get("ab", None)
            # ab
            traces += [go.Scatter3d(x=[nodes[edge[0,i_l], 0], nodes[edge[1,i_l], 0]],
                                     y=[nodes[edge[0,i_l], 1], nodes[edge[1,i_l], 1]],
                                     z=[nodes[edge[0,i_l], 2], nodes[edge[1,i_l], 2]],
                                     showlegend=False, mode='lines',
                                     line=dict(
                                         color='rgb(125,125,125)', dash='dash', width=1, showscale=False
                                     )) for i_l in range(edge.shape[1])
                        ]
            traces.append(go.Scatter3d(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode='markers',
                                       marker=dict(
                                           size=size, color=color, coloraxis="coloraxis", showscale=False,
                                       ),line={"color":'rgb(125,125,125)',"width": 0.5},))
        nodes = v[i_v].get("ag", None)
        if nodes is not None:
            color = cols[i_v].get("ag", None)
            edge = edges[i_v].get("ag", None)
            # ab
            traces += [go.Scatter3d(x=[nodes[edge[0,i_l], 0], nodes[edge[1,i_l], 0]],
                                     y=[nodes[edge[0,i_l], 1], nodes[edge[1,i_l], 1]],
                                     z=[nodes[edge[0,i_l], 2], nodes[edge[1,i_l], 2]],
                                     showlegend=False, mode='lines',
                                     line=dict(
                                         color='rgb(125,125,125)', dash='dash', width=1, showscale=False
                                     )) for i_l in range(edge.shape[1])
                        ]
            traces.append(go.Scatter3d(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode='markers',
                                       marker=dict(
                                           size=size, color=color, coloraxis="coloraxis2", showscale=False,
                                       ),line={"color":'rgb(125,125,125)',"width": 0.5}))

        fig.add_traces(traces, rows=1, cols=i_v + 1)

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None],
                                  mode='markers', marker=dict(
            # colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, coloraxis="coloraxis",
            colorbar=dict(thickness=10, outlinewidth=0)
        ), hoverinfo='none')
    fig['layout']['showlegend'] = False
    fig.add_traces([colorbar_trace, ], rows=1, cols=i_v + 1)

    if show:
        fig.show()
    return fig


def plot_meshes_over_pc(v, f, v_pc, cols, cols_pc, subplots_titles, colorscale='Plasma', size=5, opacity=0.2,
                        show=True):
    """     Plot 3D graphs with Plotly

    :param v:   list of graphs
    :param f:   list of faces
    :param v_pc:    list of point clouds
    :param cols:    list of colors for each graph
    :param cols_pc: list of colors for each point cloud
    :param subplots_titles:     titles for the subplots
    :param colorscale:  colorscale for the meshes
    :param size:    size of the points
    :param opacity: opacity of the meshes
    :param show:    show plot
    :return:    fig
    """

    fig = make_subplots(
        rows=1, cols=len(v),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'} for i in range(len(v))]])

    fig.layout.coloraxis.colorscale = colorscale
    for i in range(len(v)):
        fig.layout[f"scene{i + 1}"].update(dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
        ))
    for i_v in range(len(v)):
        fig.add_traces([go.Mesh3d(x=v[i_v][:, 0], y=v[i_v][:, 1], z=v[i_v][:, 2],
                                  i=f[i_v][:, 0], j=f[i_v][:, 1], k=f[i_v][:, 2],
                                  contour=go.mesh3d.Contour(color="#101010", width=2, show=True), opacity=opacity,
                                  intensity=cols[i_v], coloraxis="coloraxis", showscale=False,
                                  # cmax=cmax, cmin=cmin, colorscale=colorscale,
                                  ),
                        go.Scatter3d(x=v_pc[i_v][:, 0], y=v_pc[i_v][:, 1], z=v_pc[i_v][:, 2],
                                     mode='markers',
                                     marker=dict(
                                         size=size, color=cols_pc[i_v],
                                         coloraxis="coloraxis", showscale=False,
                                         # cmax=cmax, cmin=cmin, colorscale=colorscale,
                                     )),
                        ], rows=1,
                       cols=i_v + 1)

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None],
                                  mode='markers', marker=dict(
            # colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, coloraxis="coloraxis",
            colorbar=dict(thickness=10, outlinewidth=0)
        ), hoverinfo='none')
    fig['layout']['showlegend'] = False
    fig.add_traces([colorbar_trace, ], rows=1, cols=i_v + 1)

    if show:
        fig.show()
    return fig

def plot_abag_meshes_over_pc(v, f, v_pc, cols, cols_pc, subplots_titles, colorscale='Plasma', size=5, opacity=0.2,
                        show=True, x_eye = -1.25, y_eye = 2, z_eye = 0.5, cmax=1, cmin=-0.4):
    """     Plot 3D graphs with Plotly

    :param v:   list of graphs
    :param f:   list of faces
    :param v_pc:    list of point clouds
    :param cols:    list of colors for each graph
    :param cols_pc: list of colors for each point cloud
    :param subplots_titles:     titles for the subplots
    :param colorscale:  colorscale for the meshes
    :param size:    size of the points
    :param opacity: opacity of the meshes
    :param show:    show plot
    :return:    fig
    """

    fig = make_subplots(
        rows=1, cols=len(v),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'} for i in range(len(v))]])

    fig.update_layout(coloraxis={'colorscale': 'Blues', 'cmax':cmax, 'cmin': cmin}) # ab
    fig.update_layout(coloraxis2={'colorscale': 'Reds', 'cmax':cmax, 'cmin': cmin}) # ag
    for i in range(len(v)):
        fig.layout[f"scene{i + 1}"].update(dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
            camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        ))
    for i_v in range(len(v)):
        traces = list()
        mesh = v[i_v].get("ab", None)
        if mesh is not None:
            faces = f[i_v].get("ab", None)
            color = cols[i_v].get("ab", None)
            color_pc = cols_pc[i_v].get("ab", None)
            nodes = v_pc[i_v].get("ab", None)
            # ab
            traces.append(go.Mesh3d(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                    contour=go.mesh3d.Contour(color="#101010", width=2, show=True), opacity=opacity,
                                    intensity=color, coloraxis="coloraxis", showscale=False,
                                    ))
            traces.append(go.Scatter3d(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode='markers',
                                       marker=dict(
                                           size=size, color=color_pc, coloraxis="coloraxis", showscale=False,
                                       ),line={"color":'rgb(125,125,125)',"width": 0.5}))
        mesh = v[i_v].get("ag", None)
        if mesh is not None:
            faces = f[i_v].get("ag", None)
            color = cols[i_v].get("ag", None)
            color_pc = cols_pc[i_v].get("ag", None)
            nodes = v_pc[i_v].get("ag", None)
            # ag
            traces.append(go.Mesh3d(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                    contour=go.mesh3d.Contour(color="#101010", width=2, show=True), opacity=opacity,
                                    intensity=color, coloraxis="coloraxis2", showscale=False,
                                    ))
            traces.append(go.Scatter3d(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode='markers',
                                       marker=dict(
                                           size=size, color=color_pc, coloraxis="coloraxis2", showscale=False,
                                       ),line={"color":'rgb(125,125,125)',"width": 0.5}))


        fig.add_traces(traces, rows=1, cols=i_v + 1)

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None],
                                  mode='markers', marker=dict(
            # colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, coloraxis="coloraxis",
            colorbar=dict(thickness=10, outlinewidth=0)
        ), hoverinfo='none')
    fig['layout']['showlegend'] = False
    fig.add_traces([colorbar_trace, ], rows=1, cols=i_v + 1)

    if show:
        fig.show()
    return fig

def plot_meshes(v, f, cols, subplots_titles, colorscale='Plasma', show=True):
    """     Plot 3D graphs with Plotly

    :param v:   list of graphs
    :param f:   list of faces
    :param cols:    list of colors for each graph
    :param subplots_titles:     titles for the subplots
    :param colorscale:  colorscale for the meshes
    :param show:    show plot
    :return:    fig
    """

    fig = make_subplots(
        rows=1, cols=len(v),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'} for i in range(len(v))]])

    fig.layout.coloraxis.colorscale = colorscale
    for i in range(len(v)):
        fig.layout[f"scene{i + 1}"].update(dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
        ))
    for i_v in range(len(v)):
        fig.add_traces([go.Mesh3d(x=v[i_v][:, 0], y=v[i_v][:, 1], z=v[i_v][:, 2],
                                  i=f[i_v][:, 0], j=f[i_v][:, 1], k=f[i_v][:, 2],
                                  contour=go.mesh3d.Contour(color="#101010", width=2, show=True), opacity=1,
                                  intensity=cols[i_v], coloraxis="coloraxis", showscale=False,
                                  # cmax=cmax, cmin=cmin, colorscale=colorscale,
                                  )], rows=1,
                       cols=i_v + 1)

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None],
                                  mode='markers', marker=dict(
            # colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, coloraxis="coloraxis",
            colorbar=dict(thickness=10, outlinewidth=0)
        ), hoverinfo='none')
    fig['layout']['showlegend'] = False
    fig.add_traces([colorbar_trace, ], rows=1, cols=i_v + 1)

    if show:
        fig.show()
    return fig

def plot_abag_meshes(v, f, cols, subplots_titles, show=True, x_eye = -1.25, y_eye = 2, z_eye = 0.5):
    """     Plot 3D graphs with Plotly

    :param v:   list of graphs
    :param f:   list of faces
    :param cols:    list of colors for each graph
    :param subplots_titles:     titles for the subplots
    :param show:    show plot
    :return:    fig
    """

    fig = make_subplots(
        rows=1, cols=len(v),
        subplot_titles=subplots_titles,
        specs=[[{'type': 'scene'} for i in range(len(v))]])

    fig.update_layout(coloraxis={'colorscale': 'Blues', 'cmax':1, 'cmin': -0.4}) # ab
    fig.update_layout(coloraxis2={'colorscale': 'Reds', 'cmax':1, 'cmin': -0.4}) # ag

    for i in range(len(v)):
        fig.layout[f"scene{i + 1}"].update(dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
            camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        ))
    for i_v in range(len(v)):
        traces = list()
        mesh = v[i_v].get("ab", None)
        if mesh is not None:
            faces = f[i_v].get("ab", None)
            color = cols[i_v].get("ab", None)
            # ab
            traces.append(go.Mesh3d(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2],
                                  i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                  contour=go.mesh3d.Contour(color="#101010", width=2, show=True), opacity=1,
                                  intensity=color, coloraxis="coloraxis", showscale=False,
                                  ))
        mesh = v[i_v].get("ag", None)
        if mesh is not None:
            faces = f[i_v].get("ag", None)
            color = cols[i_v].get("ag", None)
            # ab
            traces.append(go.Mesh3d(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2],
                                  i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                  contour=go.mesh3d.Contour(color="#101010", width=2, show=True), opacity=1,
                                  intensity=color, coloraxis="coloraxis2", showscale=False,
                                  ))
        fig.add_traces(traces, rows=1,  cols=i_v + 1)

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None],
                                  mode='markers', marker=dict(
            # colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, coloraxis="coloraxis",
            colorbar=dict(thickness=10, outlinewidth=0)
        ), hoverinfo='none')
    fig['layout']['showlegend'] = False
    fig.add_traces([colorbar_trace, ], rows=1, cols=i_v + 1)

    if show:
        fig.show()
    return fig

