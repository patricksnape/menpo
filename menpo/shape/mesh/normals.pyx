import numpy as np
import cython
cimport numpy as np
cimport cython


ctypedef fused uint64_t:
    np.int32_t
    np.int64_t
    np.uint32_t
    np.uint64_t

ctypedef fused DOUBLE_TYPES:
    double
    float


cdef np.ndarray[DOUBLE_TYPES, ndim=2] normalise(np.ndarray[DOUBLE_TYPES, ndim=2] vec):
    """
    Normalise the given array of vectors.

    Parameters
    ----------
    vec : (N, 3) c-contiguous double ndarray
        The array of vectors to normalise

    Returns
    -------
    normalised : (N, 3) c-contiguous double ndarray
        Normalised array of vectors.
    """
    # Avoid divisions by almost 0 numbers
    # np.spacing(1) is equivalent to Matlab's eps
    cdef np.ndarray[DOUBLE_TYPES, ndim=1] d = np.sqrt(np.sum(vec ** 2, axis=1))
    d[d < np.spacing(1)] = 1.0
    return vec / d[..., None]


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


cpdef compute_normals(np.ndarray[DOUBLE_TYPES, ndim=2] vertex,
                      np.ndarray[uint64_t, ndim=2] face):
    """
    Compute the per-vertex and per-face normal of the vertices given a list of
    faces. Ensures that all the normals are pointing in a consistent direction
    (to avoid 'inverted' normals).

    Parameters
    ----------
    vertex : (N, 3) c-contiguous double ndarray
        The list of points to compute normals for.
    face : (M, 3) c-contiguous double ndarray
        The list of faces (triangle list).

    Returns
    -------
    vertex_normal : (N, 3) c-contiguous double ndarray
        The normal per vertex.
    face_normal : (M, 3) c-contiguous double ndarray
        The normal per face.
    """
    # Maintan the vertex dtype
    dtype = vertex.dtype
    cdef:
        uint64_t f0 = 0, f1 = 0, f2 = 0, i = 0
        uint64_t nface = face.shape[0]
        uint64_t nvert = vertex.shape[0]
        np.ndarray[DOUBLE_TYPES, ndim=2] face_normal = np.empty_like(vertex, dtype=dtype)
        np.ndarray[DOUBLE_TYPES, ndim=2] vertex_normal = np.zeros([nvert, 3], dtype=dtype)
        DOUBLE_TYPES[:, :] a, b, c

    # Calculate the cross product (per-face normal)
    # Due to the way that Cython performs type inference for fused_types, we
    # need to 'cast' the numpy arrays to buffer interfaces so that the cross
    # method can compile. For some reason Cython seems unable to do the type
    # inference directly on numpy arrays. Inline casting does not work as this
    # is a compile time issue
    a = vertex[face[:, 1], :] - vertex[face[:, 0], :]
    b = vertex[face[:, 2], :] - vertex[face[:, 0], :]
    c = face_normal
    cross(a, b, c)

    # Calculate per-vertex normal
    for i in range(nface):
        f0 = face[i, 0]
        f1 = face[i, 1]
        f2 = face[i, 2]
        for j in range(3):
            vertex_normal[f0, j] += face_normal[i, j]
            vertex_normal[f1, j] += face_normal[i, j]
            vertex_normal[f2, j] += face_normal[i, j]

    return normalise(vertex_normal), normalise(face_normal)
