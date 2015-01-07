import numpy as np
import cython
cimport numpy as np
cimport cython


ctypedef np.uint32_t uint32_t
ctypedef fused DOUBLE_TYPES:
    double
    float


cdef void normalise_inplace(DOUBLE_TYPES[:, :] vec):
    """
    Normalise the given array of vectors in-place.

    Parameters
    ----------
    vec : (N, 3) c-contiguous double ndarray
        The array of vectors to normalise.
    """
    # Non-copy cast to numpy array so we can do lazy smart numpy operations
    # on the buffer
    cdef:
        np.ndarray[DOUBLE_TYPES, ndim=2] np_vec = np.asarray(vec)
        np.ndarray[DOUBLE_TYPES, ndim=1] d = np.sqrt(np.sum(np_vec ** 2, axis=1))

    # Avoid divisions by almost 0 numbers
    # np.spacing(1) is equivalent to Matlab's eps
    d[d < np.spacing(1)] = 1.0
    np_vec /= d[..., None]


cdef inline void cross(const DOUBLE_TYPES[:, :] x, const DOUBLE_TYPES[:, :] y,
                       DOUBLE_TYPES[:, :] out):
    """
    The N x 3 cross product (returns the vectors orthogonal
    to ``x`` and ``y``). This performs the cross product on each (3, 1) vector
    in the two arrays. Assumes ``x``, ``y`` and ``out`` have the same shape.

    Parameters
    ----------
    x : (N, 3) DOUBLE_TYPES memory view
        First array to perform cross product with.
    y : (N, 3) DOUBLE_TYPES memory view
        Second array to perform cross product with.
    out : (N, 3) c-contiguous DOUBLE_TYPES ndarray
        The array of vectors representing the cross product between each
        corresponding vector.
    """
    cdef:
        size_t n = x.shape[0], i = 0
    for i in range(n):
        out[i, 0] = x[i, 1] * y[i, 2] - x[i, 2] * y[i, 1]
        out[i, 1] = x[i, 2] * y[i, 0] - x[i, 0] * y[i, 2]
        out[i, 2] = x[i, 0] * y[i, 1] - x[i, 1] * y[i, 0]


# We need explicit number arrays here because we use fancy indexing
cpdef compute_normals(np.ndarray[DOUBLE_TYPES, ndim=2] vertex,
                      np.ndarray[uint32_t, ndim=2] face):
    """
    Compute the per-vertex and per-face normal of the vertices given a list of
    faces.

    Parameters
    ----------
    vertex : (N, 3) c-contiguous DOUBLE_TYPES ndarray
        The list of points to compute normals for.
    face : (M, 3) c-contiguous DOUBLE_TYPES ndarray
        The list of faces (triangle list).

    Returns
    -------
    vertex_normal : (N, 3) c-contiguous DOUBLE_TYPES ndarray
        The normal per vertex.
    face_normal : (M, 3) c-contiguous DOUBLE_TYPES ndarray
        The normal per face.
    """
    # Maintan the vertex dtype
    dtype = vertex.dtype
    cdef:
        uint32_t f0 = 0, f1 = 0, f2 = 0
        size_t i = 0
        uint32_t nface = face.shape[0]
        uint32_t nvert = vertex.shape[0]
        DOUBLE_TYPES[:, :] face_normal = np.zeros([nface, 3], dtype=dtype)
        DOUBLE_TYPES[:, :] vertex_normal = np.zeros([nvert, 3], dtype=dtype)
        DOUBLE_TYPES[:, :] first_edges, second_edges

    # Calculate the cross product (per-face normal)
    # Due to the way that Cython performs type inference for fused_types, we
    # need to 'cast' the numpy arrays to buffer interfaces so that the cross
    # method can compile. For some reason Cython seems unable to do the type
    # inference directly on numpy arrays. Inline casting does not work as this
    # is a compile time issue
    first_edges = vertex[face[:, 1], :] - vertex[face[:, 0], :]
    second_edges = vertex[face[:, 2], :] - vertex[face[:, 0], :]
    cross(first_edges, second_edges, face_normal)

    normalise_inplace(face_normal)
    # Calculate per-vertex normal
    for i in range(nface):
        f0 = face[i, 0]
        f1 = face[i, 1]
        f2 = face[i, 2]
        for j in range(3):
            vertex_normal[f0, j] += face_normal[i, j]
            vertex_normal[f1, j] += face_normal[i, j]
            vertex_normal[f2, j] += face_normal[i, j]
    normalise_inplace(vertex_normal)

    return vertex_normal, face_normal
