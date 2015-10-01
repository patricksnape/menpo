from menpo.landmark import LandmarkGroup
from . import PointCloud, UndirectedGraph, DirectedGraph, Tree, TriMesh

def _get_number_of_vertices(lgroup):
    if isinstance(lgroup, LandmarkGroup):
        return lgroup.n_landmarks
    elif isinstance(lgroup, PointCloud):
        return lgroup.n_points
    else:
        raise ValueError("lgroup must be either a LandmarkGroup or a "
                         "PointCloud instance.")

def _get_star_graph_edges(vertices_list, root_vertex):
    edges = []
    for v in vertices_list:
        if v != root_vertex:
            edges.append([root_vertex, v])
    return edges

def _get_complete_graph_edges(vertices_list):
    n_vertices = len(vertices_list)
    edges = []
    for i in range(n_vertices-1):
        k = i + 1
        for j in range(k, n_vertices, 1):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            edges.append([v1, v2])
    return edges

def _get_chain_graph_edges(vertices_list, closed):
    n_vertices = len(vertices_list)
    edges = []
    for i in range(n_vertices-1):
        k = i + 1
        v1 = vertices_list[i]
        v2 = vertices_list[k]
        edges.append([v1, v2])
    if closed:
        v1 = vertices_list[-1]
        v2 = vertices_list[0]
        edges.append([v1, v2])
    return edges

def empty_graph(lgroup, graph_cls=Tree):
    # get number of vertices
    n_vertices = _get_number_of_vertices(lgroup)

    # create empty edges
    edges = None

    # return graph
    if graph_cls == Tree:
        raise ValueError("An empty graph cannot be a Tree instance.")
    elif graph_cls == UndirectedGraph or graph_cls == DirectedGraph:
        return graph_cls.init_from_edges(edges, n_vertices, skip_checks=True)
    else:
        raise ValueError("graph_cls must be either UndirectedGraph or "
                         "DirectedGraph.")

def star_graph(lgroup, root_vertex, graph_cls=Tree):
    # get number of vertices
    n_vertices = _get_number_of_vertices(lgroup)

    # create star graph edges
    edges = _get_star_graph_edges(range(n_vertices), root_vertex)

    # return graph
    if graph_cls == Tree:
        return graph_cls.init_from_edges(edges, n_vertices, root_vertex,
                                         skip_checks=True)
    elif graph_cls == UndirectedGraph or graph_cls == DirectedGraph:
        return graph_cls.init_from_edges(edges, n_vertices, skip_checks=True)
    else:
        raise ValueError("graph_cls must be UndirectedGraph, "
                         "DirectedGraph or Tree.")

def complete_graph(lgroup, graph_cls=UndirectedGraph):
    # get number of vertices
    n_vertices = _get_number_of_vertices(lgroup)

    # create complete graph edges
    edges = _get_complete_graph_edges(range(n_vertices))

    # return graph
    if graph_cls == Tree:
        raise ValueError("A complete graph cannot be a Tree instance.")
    elif graph_cls == UndirectedGraph or graph_cls == DirectedGraph:
        return graph_cls.init_from_edges(edges, n_vertices, skip_checks=True)
    else:
        raise ValueError("graph_cls must be either UndirectedGraph or "
                         "DirectedGraph.")

def chain_graph(lgroup, graph_cls=DirectedGraph, closed=False):
    # get number of vertices
    n_vertices = _get_number_of_vertices(lgroup)

    # create complete graph edges
    edges = _get_chain_graph_edges(range(n_vertices), closed=closed)

    # return graph
    if graph_cls == Tree:
        if closed:
            raise ValueError("A complete graph cannot be a Tree instance.")
        else:
            return graph_cls.init_from_edges(edges, n_vertices, root_vertex=0,
                                             skip_checks=True)
    elif graph_cls == UndirectedGraph or graph_cls == DirectedGraph:
        return graph_cls.init_from_edges(edges, n_vertices, skip_checks=True)
    else:
        raise ValueError("graph_cls must be either UndirectedGraph or "
                         "DirectedGraph.")

def delaunay_graph(lgroup):
    # get TriMesh instance that estimates the delaunay triangulation
    if isinstance(lgroup, LandmarkGroup):
        trimesh = TriMesh(lgroup.lms.points)
        n_vertices = lgroup.n_landmarks
    elif isinstance(lgroup, PointCloud):
        trimesh = TriMesh(lgroup.points)
        n_vertices = lgroup.n_points
    else:
        raise ValueError("lgroup must be either a LandmarkGroup or a "
                         "PointCloud instance.")

    # get edges
    edges = trimesh.edge_indices()

    # return graph
    return UndirectedGraph.init_from_edges(edges, n_vertices, skip_checks=True)
