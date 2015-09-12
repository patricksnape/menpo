import numpy as np
from scipy.sparse import block_diag, lil_matrix, csr_matrix

from menpo.math import as_matrix
from menpo.visualize import print_dynamic, progress_bar_str


class GMRFModel(object):
    r"""
    Trains a Gaussian Markov Random Field (GMRF).

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
        if graph.n_edges == 0:
            # no edges on the graph, so create a block diagonal precision
            self.mean_vector, self.precision = \
                _compute_block_diagonal_precision_matrix(
                    data, graph, self.n_features_per_vertex, single_precision,
                    sparse, n_components, verbose=verbose)
        else:
            # graph has edges, so create sparse precision matrix
            self.mean_vector, self.precision = _compute_sparse_precision_matrix(
                data, graph, mode, self.n_features_per_vertex,
                single_precision, sparse, n_components, verbose=verbose)

        # assign arguments
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

    def increment(self, samples, n_samples=None, forgetting_factor=1.0,
                  verbose=False):
        r"""
        Update the eigenvectors, eigenvalues and mean vector of this model
        by performing incremental PCA on the given samples.

        Parameters
        ----------
        samples : `list` of :map:`Vectorizable`
            List of new samples to update the model from.
        n_samples : `int`, optional
            If provided then ``samples``  must be an iterator that yields
            ``n_samples``. If not provided then samples has to be a
            list (so we know how large the data matrix needs to be).
        forgetting_factor : ``[0.0, 1.0]`` `float`, optional
            Forgetting factor that weights the relative contribution of new
            samples vs old samples. If 1.0, all samples are weighted equally
            and, hence, the results is the exact same as performing batch
            PCA on the concatenated list of old and new simples. If <1.0,
            more emphasis is put on the new samples. See [1] for details.

        References
        ----------
        .. [1] David Ross, Jongwoo Lim, Ruei-Sung Lin, Ming-Hsuan Yang.
           "Incremental Learning for Robust Visual Tracking". IJCV, 2007.
        """
        # build a data matrix from the new samples
        data = as_matrix(samples, length=n_samples, verbose=verbose)
        # (n_samples, n_features)
        n_new_samples = data.shape[0]

        # compute incremental pca
        e_vectors, e_values, m_vector = ipca(
            data, self._components, self._eigenvalues, self.n_samples,
            m_a=self.mean_vector, f=forgetting_factor)

        # if the number of active components is the same as the total number
        # of components so it will be after this method is executed
        reset = (self.n_active_components == self.n_components)

        # update mean, components, eigenvalues and number of samples
        self.mean_vector = m_vector
        self._components = e_vectors
        self._eigenvalues = e_values
        self.n_samples += n_new_samples

        # reset the number of active components to the total number of
        # components
        if reset:
            self.n_active_components = self.n_components

    def mahalanobis_distance(self, instance, subtract_mean=True,
                             square_root=False):
        return self.mahalanobis_distance_vector(instance.as_vector(),
                                                subtract_mean=subtract_mean,
                                                square_root=square_root)

    def mahalanobis_distance_vector(self, vector_instance, subtract_mean=True,
                                    square_root=False):
        # create data vector
        if subtract_mean:
            vector_instance = vector_instance - self.mean_vector

        # make sure we have the correct data type
        if self.sparse:
            vector_instance = csr_matrix(vector_instance)

        # compute mahalanobis
        d = vector_instance.dot(self.precision).dot(vector_instance.T)

        # if scipy.sparse, get the scalar value from the (1, 1) matrix
        if self.sparse:
            d = d[0, 0]

        # square root
        if square_root:
            return np.sqrt(d)
        else:
            return d

    def eigenanalysis(self):
        pass

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
                  ' - {}\n' \
                  ' - The data of the vertexes of each edge are {}.\n' \
                  '{}\n' \
                  ' - # variables (vertexes):  {}\n' \
                  ' - # features per variable: {}\n' \
                  ' - # features in total:     {}\n' \
                  '{}\n' \
                  ' - # samples:               {}\n'.format(
            self.graph.__str__(), mode_str, Q_str, self.graph.n_vertices,
            self.n_features_per_vertex, self.n_features, svd_str,
            self.n_samples)
        return str_out


def _compute_mean(x):
    return np.mean(x, axis=0)


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


def _compute_block_diagonal_precision_matrix(X, graph, n_features_per_vertex,
                                             single_precision, sparse,
                                             n_components, verbose=False):
    # Compute covariance matrix for each patch
    cov_list = []
    for e in range(graph.n_vertices):
        # print progress
        if verbose:
            print_dynamic('Distribution per vertex - {}'.format(
                progress_bar_str((e + 1.) / graph.n_vertices, show_bar=False)))

        # find indices in target precision matrix
        i_from = e * n_features_per_vertex
        i_to = (e + 1) * n_features_per_vertex

        # compute covariance
        edge_cov = np.cov(X[:, i_from:i_to].T)

        # invert covariance
        inv_cov = _covariance_matrix_inverse(edge_cov, n_components)

        # store covariance
        cov_list.append(inv_cov)

    # create final sparse covariance matrix
    if sparse:
        Q = block_diag(cov_list).tocsr()
    else:
        Q = block_diag(cov_list).todense()
    if single_precision:
        Q = np.require(Q, dtype=np.float32)

    return _compute_mean(X), Q


def _compute_sparse_precision_matrix(X, graph, mode, n_features_per_vertex,
                                     single_precision, sparse, n_components,
                                     verbose=False):
    # Initialize block sparse precision matrix
    Q = _initialize_precision_matrix(X.shape[1], sparse, single_precision)

    # Define function that formats the edge's data
    if mode == 'concatenation':
        generate_edge_data = lambda x1, x2: np.hstack((x1, x2))
    elif mode == 'subtraction':
        generate_edge_data = lambda x1, x2: x1 - x2
    else:
        raise ValueError("mode must be either ''concatenation'' "
                         "or ''subtraction''; {} is given.".format(mode))

    # Compute covariance matrix for each edge
    for e in range(graph.n_edges):
        # print progress
        if verbose:
            print_dynamic('Distribution per edge - {}'.format(
                progress_bar_str((e + 1.) / graph.n_edges, show_bar=False)))

        # edge vertices
        v1 = graph.edges[e, 0]
        v2 = graph.edges[e, 1]

        # find indices in target precision matrix
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex

        # data concatenation
        edge_data = generate_edge_data(X[:, v1_from:v1_to], X[:, v2_from:v2_to])

        # compute covariance
        edge_cov = np.cov(edge_data.T)

        # invert covariance
        inv_cov = _covariance_matrix_inverse(edge_cov, n_components)

        # insert to precision matrix
        if mode == 'concatenation':
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
        elif mode == 'subtraction':
            # v1, v2
            Q[v1_from:v1_to, v2_from:v2_to] = -inv_cov
            # v2, v1
            Q[v2_from:v2_to, v1_from:v1_to] = -inv_cov
            # v1, v1
            Q[v1_from:v1_to, v1_from:v1_to] += inv_cov
            # v2, v2
            Q[v2_from:v2_to, v2_from:v2_to] += inv_cov

        # convert Q to csr if sparse is enabled
        if sparse:
            return _compute_mean(X), Q.tocsr()
        else:
            return _compute_mean(X), Q
