import numpy as np
from scipy.sparse import block_diag, lil_matrix

from menpo.math import as_matrix
from menpo.visualize import print_dynamic, progress_bar_str

class GMRFModel(object):
    r"""
    A :map:`MeanInstanceLinearModel` where components are Principal
    Components.

    Principal Component Analysis (PCA) by eigenvalue decomposition of the
    data's scatter matrix. For details of the implementation of PCA, see
    :map:`pca`.

    Parameters
    ----------
    samples : `list` or `iterable` of :map:`Vectorizable`
        List or iterable of samples to build the model from.
    centre : `bool`, optional
        When ``True`` (default) PCA is performed after mean centering the data.
        If ``False`` the data is assumed to be centred, and the mean will be
        ``0``.
    n_samples : `int`, optional
        If provided then ``samples``  must be an iterator that yields
        ``n_samples``. If not provided then samples has to be a `list` (so we
        know how large the data matrix needs to be).
     """
    def __init__(self, samples, graph, mode='concatenation', n_components=None,
                 single_precision=False, sparse=True, n_samples=None,
                 verbose=False):
        # build a data matrix from all the samples
        data, self.template_instance = as_matrix(
            samples, length=n_samples, return_template=True, verbose=verbose)
        # (n_samples, n_features)
        self.n_samples = data.shape[0]
        # get n_features
        self.n_features = data.shape[1]
        self.n_features_per_vertex = self.n_features / graph.n_vertices

        # compute precision matrix
        if mode == 'concatenation':
            self.mean_vector, self.precision = _gmrf_concatenation(
                data, graph, self.n_features_per_vertex, sparse=sparse,
                n_components=n_components, single_precision=single_precision,
                verbose=verbose)
        elif mode == 'subtraction':
            self.mean_vector, self.precision = _gmrf_subtraction(
                data, graph, self.n_features_per_vertex, sparse=sparse,
                n_components=n_components, single_precision=single_precision,
                verbose=verbose)
        else:
            raise ValueError("mode must be either ''concatenation'' "
                             "or ''subtraction''; {} is given.".format(mode))

        # assign graph
        self.graph = graph
        self.mode = mode
        self.n_components = n_components
        self.sparse = sparse
        self.single_precision = single_precision

    def mean(self):
        r"""
        Return the mean of the model.

        :type: :map:`Vectorizable`
        """
        return self.template_instance.from_vector(self.mean_vector)

    def __str__(self):
        svd_str = (' - # SVD components:        {}'.format(self.n_components)
                   if self.n_components is not None else ' - No ' 'SVD used.')
        _Q_sparse = 'scipy.sparse' if self.sparse else 'numpy.array'
        _Q_precision = 'single' if self.single_precision else 'double'
        Q_str = ' - Q is stored as {} with {} precision'.format(_Q_sparse,
                                                                _Q_precision)
        mode_str = ('concatenated' if self.mode == 'concatenation' else
                    'subtracted')
        str_out = 'Gaussian MRF Model \n' \
                  ' - The data of the vertexes of each edge are {}.\n' \
                  '{}\n' \
                  ' - # variables (vertexes):  {}\n' \
                  ' - # features per variable: {}\n' \
                  ' - # features in total:     {}\n' \
                  ' - # edges:                 {}\n' \
                  '{}\n' \
                  ' - # samples:               {}\n'.format(
            mode_str, Q_str, self.graph.n_vertices, self.n_features_per_vertex,
            self.n_features, self.graph.n_edges, svd_str, self.n_samples)
        return str_out


def _gmrf_mean(X):
    return np.mean(X, axis=0)


def _initialize_precision_matrix(n_features, sparse, single_precision):
    if single_precision:
        if sparse:
            return lil_matrix((n_features, n_features), dtype=np.float32)
        else:
            return np.zeros((n_features, n_features), dtype=np.float32)
    else:
        if sparse:
            return lil_matrix((n_features, n_features))
        else:
            return np.zeros((n_features, n_features))


def _covariance_matrix_inverse(cov_mat, n_components):
    if n_components is None:
        return np.linalg.inv(cov_mat)
    else:
        try:
            s, v, d = np.linalg.svd(cov_mat)
            s = s[:, :n_components]
            v = v[:n_components]
            d = d[:n_components, :]
            return s.dot(np.diag(1/v)).dot(d)
        except:
            return np.linalg.inv(cov_mat)


def _gmrf_concatenation(X, graph, n_features_per_vertex, single_precision,
                        n_components, sparse, verbose=False):
    n_samples, n_features = X.shape

    # Initialize block sparse precision matrix
    Q = _initialize_precision_matrix(n_features, sparse, single_precision)

    # Compute covariance matrix for each edge
    for e in range(graph.n_edges):
        # print progress
        if verbose:
            print_dynamic('Distribution per edge - {}'.format(
                progress_bar_str((e + 1.) / graph.n_edges, show_bar=False)))

        # edge vertices
        v1 = np.min(graph.edges[e, :])
        v2 = np.max(graph.edges[e, :])

        # find indices in target precision matrix
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex

        # data concatenation
        edge_data = np.hstack((X[:, v1_from:v1_to], X[:, v2_from:v2_to]))

        # compute covariance
        cov_v1_v2 = np.linalg.inv(np.cov(edge_data.T))

        # invert covariance
        inv_cov = _covariance_matrix_inverse(cov_v1_v2, n_components)

        # v1, v2
        Q[v1_from:v1_to, v2_from:v2_to] = inv_cov[:n_features_per_vertex,
                                                  n_features_per_vertex::]

        # v2, v1
        Q[v2_from:v2_to, v1_from:v1_to] = inv_cov[n_features_per_vertex::,
                                                  :n_features_per_vertex]

        # v1, v1
        Q[v1_from:v1_to, v1_from:v1_to] += inv_cov[:n_features_per_vertex,
                                                   :n_features_per_vertex]

        # v2, v2
        Q[v2_from:v2_to, v2_from:v2_to] += inv_cov[n_features_per_vertex::,
                                                   n_features_per_vertex::]

    return _gmrf_mean(X), Q


def _gmrf_subtraction(X, graph, n_features_per_vertex, single_precision,
                      n_components, sparse, verbose=False):
    n_samples, n_features = X.shape

    # Initialize block sparse precision matrix
    Q = _initialize_precision_matrix(n_features, sparse, single_precision)

    # Compute covariance matrix for each edge
    for e in range(graph.n_edges):
        # print progress
        if verbose:
            print_dynamic('Distribution per edge - {}'.format(
                progress_bar_str(float(e + 1) / graph.n_edges, show_bar=False)))

        # edge vertices
        v1 = graph.edges[e, 0]
        v2 = graph.edges[e, 1]

        # find indices in target precision matrix
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex

        # data subtraction
        edge_data = X[:, v1_from:v1_to] - X[:, v2_from:v2_to]

        # compute covariance
        cov_v1_v2 = np.linalg.inv(np.cov(edge_data.T))

        # invert covariance
        inv_cov = _covariance_matrix_inverse(cov_v1_v2, n_components)

        # v1, v2
        Q[v1_from:v1_to, v2_from:v2_to] = -inv_cov

        # v2, v1
        Q[v2_from:v2_to, v1_from:v1_to] = -inv_cov

        # v1, v1
        Q[v1_from:v1_to, v1_from:v1_to] += inv_cov

        # v2, v2
        Q[v2_from:v2_to, v2_from:v2_to] += inv_cov

    return _gmrf_mean(X), Q
