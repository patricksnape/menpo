from menpo.math import as_matrix

class GMRFModel(object):

    def __init__(self, samples, graph, variable_length, centre=True,
                 n_samples=None, verbose=False):
        # build a data matrix from all the samples
        data, template = as_matrix(samples, length=n_samples,
                                   return_template=True, verbose=verbose)
        # (n_samples, n_features)
        self.n_samples = data.shape[0]

        # compute pca
        e_vectors, e_values, mean = pca(data, centre=centre, inplace=True)

        super(PCAModel, self).__init__(e_vectors, mean, template)
        self.centred = centre
        self._eigenvalues = e_values
        # start the active components as all the components
        self._n_active_components = int(self.n_components)
        self._trimmed_eigenvalues = np.array([])

    def __str__(self):
        str_out = 'PCA Model \n' \
                  ' - centred:              {}\n' \
                  ' - # features:           {}\n' \
                  ' - # active components:  {}\n' \
                  ' - kept variance:        {:.2}  {:.1%}\n' \
                  ' - noise variance:       {:.2}  {:.1%}\n' \
                  ' - total # components:   {}\n' \
                  ' - components shape:     {}\n'.format(
            self.centred,  self.n_features, self.n_active_components,
            self.variance(), self.variance_ratio(), self.noise_variance(),
            self.noise_variance_ratio(), self.n_components,
            self.components.shape)
        return str_out