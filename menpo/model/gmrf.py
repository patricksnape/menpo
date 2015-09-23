import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from menpo.math import as_matrix
from menpo.visualize import print_progress, bytes_str


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
    bias : int, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
     """
    def __init__(self, samples, graph, mode='concatenation', n_components=None,
                 single_precision=False, sparse=True, n_samples=None, bias=0,
                 incremental=False, verbose=False):
        # Build a data matrix from all the samples
        data, self.template_instance = as_matrix(
            samples, length=n_samples, return_template=True, verbose=verbose)
        # (n_samples, n_features)
        self.n_samples = data.shape[0]
        # n_features and n_features_per_vertex
        self.n_features = data.shape[1]
        self.n_features_per_vertex = self.n_features / graph.n_vertices

        # Assign arguments
        self.graph = graph
        self.mode = mode
        self.n_components = n_components
        self.sparse = sparse
        self.single_precision = single_precision
        self.bias = bias
        self.is_incremental = incremental

        # Compute mean vector
        self.mean_vector = np.mean(data, axis=0)

        # Compute precision matrix
        # First, initialize block sparse precision matrix.
        self.precision = self._initialize_precision_matrix(verbose=verbose)
        # Then, compute covariance matrix of each edge.
        if self.graph.n_edges == 0:
            # no edges on the graph, so create a block diagonal precision
            cov_list = _vertices_covariance_matrices(
                data, graph, self.n_features_per_vertex, bias=bias,
                verbose=verbose)
            self._set_block_diagonal_precision_matrix(
                covariance_matrices=cov_list, verbose=verbose)
        else:
            # Graph has edges, so create sparse precision matrix.
            cov_list = _edges_covariance_matrices(
                data, graph, self.n_features_per_vertex, mode=mode, bias=bias,
                verbose=verbose)
            # Finally, invert and store covariances to precision matrix.
            self._set_block_sparse_precision_matrix(
                covariance_matrices=cov_list, verbose=verbose)

        # If sparse is enabled, it is better to convert self.precision from
        # lil_matrix to csr_matrix
        if self.sparse:
            self.precision = self.precision.tocsr()

        # If incremental flag is enabled, store the covariance matrices
        self._covariance_matrices = None
        if self.is_incremental:
            self._covariance_matrices = cov_list

    def mean(self):
        r"""
        Return the mean of the model.

        :type: :map:`Vectorizable`
        """
        return self.template_instance.from_vector(self.mean_vector)

    def increment(self, samples, n_samples=None, verbose=False):
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
        # Check if it can be incrementally updated
        if not self.is_incremental:
            raise ValueError('GMRF cannot be incrementally updated.')

        # Reset the values of the precision matrix
        if self.sparse:
            self.precision[self.precision.nonzero()] = 0.
        else:
            self.precision[:] = 0.

        # Build a data matrix from the new samples
        data = as_matrix(samples, length=n_samples, verbose=verbose)

        # Update precision matrix
        if self.graph.n_edges == 0:
            # No edges on the graph, so update a block diagonal precision
            _increment_vertices_covariance_matrices(
                data, self.graph, self.mean_vector, self._covariance_matrices,
                self.n_samples, self.n_features_per_vertex, bias=self.bias,
                verbose=verbose)
            self._set_block_diagonal_precision_matrix(
                covariance_matrices=None, verbose=verbose)
        else:
            # Graph has edges, so update sparse precision matrix
            _increment_edges_covariance_matrices(
                data, self.graph, self.mean_vector, self._covariance_matrices,
                self.n_samples, self.n_features_per_vertex,
                mode=self.mode, bias=self.bias, verbose=verbose)
            self._set_block_sparse_precision_matrix(
                covariance_matrices=None, verbose=verbose)

        # Update mean and number of samples
        self.mean_vector = _increment_multivariate_gaussian_mean(
            data, self.mean_vector, self.n_samples)
        self.n_samples += data.shape[0]

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

    def _initialize_precision_matrix(self, verbose=False):
        # select data type
        if self.single_precision:
            dtype = np.float32
        else:
            dtype = np.float64

        # create matrix based on sparsity
        if self.sparse:
            precision = lil_matrix((self.n_features, self.n_features),
                                   dtype=dtype)
        else:
            precision = np.zeros((self.n_features, self.n_features),
                                 dtype=dtype)

        # print info
        if verbose and not self.sparse:
            print('Allocated precision matrix of size {}'.format(
                bytes_str(precision.nbytes)))
        return precision

    def _set_block_sparse_precision_matrix(self, covariance_matrices=None,
                                           verbose=False):
        # Print information if asked
        if verbose:
            edges = print_progress(
                range(self.graph.n_edges), n_items=self.graph.n_edges,
                prefix='Building precision matrix')
        else:
            edges = range(self.graph.n_edges)

        # Store inverse of covariance matrix for each edge
        for e in edges:
            # edge vertices
            v1 = self.graph.edges[e, 0]
            v2 = self.graph.edges[e, 1]

            # find indices in target precision matrix
            v1_from = v1 * self.n_features_per_vertex
            v1_to = (v1 + 1) * self.n_features_per_vertex
            v2_from = v2 * self.n_features_per_vertex
            v2_to = (v2 + 1) * self.n_features_per_vertex

            # invert covariance matrix
            if covariance_matrices is None:
                inv_cov = _covariance_matrix_inverse(
                    self._covariance_matrices[e], self.n_components)
            else:
                inv_cov = _covariance_matrix_inverse(covariance_matrices[e],
                                                     self.n_components)

            # insert to precision matrix
            if self.mode == 'concatenation':
                # v1, v2
                self.precision[v1_from:v1_to, v2_from:v2_to] = \
                    inv_cov[:self.n_features_per_vertex,
                            self.n_features_per_vertex::]
                # v2, v1
                self.precision[v2_from:v2_to, v1_from:v1_to] = \
                    inv_cov[self.n_features_per_vertex::,
                            :self.n_features_per_vertex]
                # v1, v1
                self.precision[v1_from:v1_to, v1_from:v1_to] += \
                    inv_cov[:self.n_features_per_vertex,
                            :self.n_features_per_vertex]
                # v2, v2
                self.precision[v2_from:v2_to, v2_from:v2_to] += \
                    inv_cov[self.n_features_per_vertex::,
                            self.n_features_per_vertex::]
            elif self.mode == 'subtraction':
                # v1, v2
                self.precision[v1_from:v1_to, v2_from:v2_to] = -inv_cov
                # v2, v1
                self.precision[v2_from:v2_to, v1_from:v1_to] = -inv_cov
                # v1, v1
                self.precision[v1_from:v1_to, v1_from:v1_to] += inv_cov
                # v2, v2
                self.precision[v2_from:v2_to, v2_from:v2_to] += inv_cov

    def _set_block_diagonal_precision_matrix(self, covariance_matrices=None,
                                             verbose=False):
        # Print information if asked
        if verbose:
            vertices = print_progress(
                range(self.graph.n_vertices), n_items=self.graph.n_vertices,
                prefix='Building precision matrix')
        else:
            vertices = range(self.graph.n_vertices)

        # Invert and store covariance matrix for each patch
        for v in vertices:
            # find indices in target precision matrix
            i_from = v * self.n_features_per_vertex
            i_to = (v + 1) * self.n_features_per_vertex

            # invert covariance matrix
            if covariance_matrices is None:
                inv_cov = _covariance_matrix_inverse(
                    self._covariance_matrices[v], self.n_components)
            else:
                inv_cov = _covariance_matrix_inverse(covariance_matrices[v],
                                                     self.n_components)

            # insert to precision matrix
            self.precision[i_from:i_to, i_from:i_to] = inv_cov

    def __str__(self):
        incremental_str = (' - Can be incrementally updated.' if
                           self.is_incremental else ' - Cannot be '
                                                    'incrementally updated.')
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
                  ' - # samples:               {}\n' \
                  '{}\n'.format(
            self.graph.__str__(), mode_str, Q_str, self.graph.n_vertices,
            self.n_features_per_vertex, self.n_features, svd_str,
            self.n_samples, incremental_str)
        return str_out


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


def _edges_covariance_matrices(X, graph, n_features_per_vertex,
                               mode='concatenation', bias=0, verbose=False):
    # Define function that formats the edge's data
    if mode == 'concatenation':
        generate_edge_data = lambda x1, x2: np.hstack((x1, x2))
    elif mode == 'subtraction':
        generate_edge_data = lambda x1, x2: x1 - x2
    else:
        raise ValueError("mode must be either ''concatenation'' "
                         "or ''subtraction''; {} is given.".format(mode))

    # Print information if asked
    if verbose:
        edges = print_progress(
            graph.edges, n_items=graph.n_edges, prefix='Covariance per edge',
            end_with_newline=False)
    else:
        edges = graph.edges

    # Compute covariance matrix for each edge
    cov_list = []
    for e in edges:
        # edge vertices
        v1 = e[0]
        v2 = e[1]

        # find indices in target precision matrix
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex

        # data concatenation
        edge_data = generate_edge_data(X[:, v1_from:v1_to], X[:, v2_from:v2_to])

        # compute covariance matrix and store it
        cov_list.append(np.cov(edge_data, rowvar=0, bias=bias))

    return cov_list


def _vertices_covariance_matrices(X, graph, n_features_per_vertex, bias=0,
                                  verbose=False):
    # Print information if asked
    if verbose:
        vertices = print_progress(
            range(graph.n_vertices), n_items=graph.n_vertices,
            prefix='Covariance per vertex', end_with_newline=False)
    else:
        vertices = range(graph.n_vertices)

    # Compute covariance matrix for each patch
    cov_list = []
    for v in vertices:
        # find indices in target precision matrix
        i_from = v * n_features_per_vertex
        i_to = (v + 1) * n_features_per_vertex

        # compute covariance
        cov_list.append(np.cov(X[:, i_from:i_to], rowvar=0, bias=bias))

    return cov_list


def _increment_edges_covariance_matrices(X, graph, m, cov_list, n,
                                         n_features_per_vertex,
                                         mode='concatenation', bias=0,
                                         verbose=False):
    # Define function that formats the edge's data
    if mode == 'concatenation':
        generate_edge_data = lambda x1, x2: np.hstack((x1, x2))
    elif mode == 'subtraction':
        generate_edge_data = lambda x1, x2: x1 - x2
    else:
        raise ValueError("mode must be either ''concatenation'' "
                         "or ''subtraction''; {} is given.".format(mode))

    # Print information if asked
    if verbose:
        edges = print_progress(
            graph.edges, n_items=graph.n_edges, prefix='Covariance per edge',
            end_with_newline=False)
    else:
        edges = graph.edges

    # Increment covariance matrix for each edge
    for i, e in enumerate(edges):
        # edge vertices
        v1 = e[0]
        v2 = e[1]

        # find indices in target precision matrix
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex

        # data concatenation
        edge_data = generate_edge_data(X[:, v1_from:v1_to], X[:, v2_from:v2_to])

        # get mean vector
        mean_vector = generate_edge_data(m[v1_from:v1_to], m[v2_from:v2_to])

        # increment
        _, cov_list[i] = _increment_multivariate_gaussian_cov(
            edge_data, mean_vector, cov_list[i], n, bias=bias)


def _increment_vertices_covariance_matrices(X, graph, m, cov_list, n,
                                            n_features_per_vertex, bias=0,
                                            verbose=False):
    # Print information if asked
    if verbose:
        vertices = print_progress(
            range(graph.n_vertices), n_items=graph.n_vertices,
            prefix='Covariance per vertex', end_with_newline=False)
    else:
        vertices = range(graph.n_vertices)

    # Compute covariance matrix for each patch
    for v in vertices:
        # find indices in target precision matrix
        i_from = v * n_features_per_vertex
        i_to = (v + 1) * n_features_per_vertex

        # compute covariance
        edge_data = X[:, i_from:i_to]
        mean_vector = m[i_from:i_to]
        _, cov_list[v] = _increment_multivariate_gaussian_cov(
            edge_data, mean_vector, cov_list[v], n, bias=bias)


def _increment_multivariate_gaussian_mean(X, m, n):
    # Get new number of samples
    new_n = X.shape[0]

    # Update mean vector
    # m_{new} = (n m + \sum_{i=1}^{n_{new}} x_i) / (n + n_{new})
    # where: m       -> old mean vector
    #        n_{new} -> new number of samples
    #        n       -> old number of samples
    #        x_i     -> new data vectors
    return (n * m + np.sum(X, axis=0)) / (n + new_n)


def _increment_multivariate_gaussian_cov(X, m, S, n, bias=0):
    # Get new number of samples
    new_n = X.shape[0]

    # Update mean vector
    # m_{new} = (n m + \sum_{i=1}^{n_{new}} x_i) / (n + n_{new})
    # where: m_{new} -> new mean vector
    #        m       -> old mean vector
    #        n_{new} -> new number of samples
    #        n       -> old number of samples
    #        x_i     -> new data vectors
    new_m = _increment_multivariate_gaussian_mean(X, m, n)

    # Select the normalization value
    if bias == 1:
        k = n
    elif bias == 0:
        k = n - 1
    else:
        raise ValueError("bias must be either 0 or 1")

    # Update covariance matrix
    # S__{new} = (k S + n m^T m + X^T X - (n + n_{new}) m_{new}^T m_{new})
    #                                                            / (k + n_{new})
    m1 = n * m[None, :].T.dot(m[None, :])
    m2 = (n + new_n) * new_m[None, :].T.dot(new_m[None, :])
    new_S = (k * S + m1 + X.T.dot(X) - m2) / (k + new_n)

    return new_m, new_S

