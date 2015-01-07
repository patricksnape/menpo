# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from menpo.cy_utils cimport dtype_from_memoryview


ctypedef fused IMAGE_TYPES:
    float
    double


cdef extern from "cpp/central_difference.h":
    void central_difference[T](const T* input, const size_t rows,
                               const size_t cols, const size_t n_channels,
                               T* output)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gradient_cython(IMAGE_TYPES[:, :, :] input):

    cdef:
        size_t n_channels_in = input.shape[0]
        size_t rows = input.shape[1]
        size_t cols = input.shape[2]
        size_t n_channels_out = n_channels_in * 2

    # Maintain the dtype that was passed in (float or double)
    dtype = dtype_from_memoryview(input)
    cdef IMAGE_TYPES[:, :, :] output = np.zeros((n_channels_out, rows, cols),
                                                dtype=dtype)

    central_difference(&input[0, 0, 0], rows, cols, n_channels_in,
                       &output[0, 0, 0])

    # As numpy array without copying
    return np.asarray(output)
