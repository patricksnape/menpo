import numpy as np
cimport numpy as cnp


cdef extern from "./fastpwa/opcode/Opcode.h" namespace "Opcode":
    cdef cppclass Model:
        Model()

cdef extern from "./fastpwa/opcode_wrapper.h":
    Model* createOpcode(const float* vertices, const int n_v,
                        const int* faces, const int n_f);

    void intersect(const Model* opcode, const float* rayo, const int n_c,
                   int* indices, float* alphas, float* betas);

cdef class OpcodePWA:
    cdef Model* opcode_model
    cdef object points
    cdef object trilist

    def __cinit__(self, float[:, ::1] points not None,
                  int[:, ::1] trilist not None):
        self.trilist = trilist
        self.points = points
        self.opcode_model = createOpcode(&points[0, 0],
                                         points.shape[0],
                                         &trilist[0, 0],
                                         trilist.shape[0])

    def __dealloc__(self):
        del self.opcode_model

    def __reduce__(self):
        r"""
        Implement the reduction protocol so this object is copyable/picklable
        """
        return self.__class__, (np.asarray(self.points),
                                np.asarray(self.trilist))

    def index_alpha_beta(self, float[:, ::1] points not None):
        # create three c numpy arrays for storing our output into
        cdef cnp.ndarray[float, ndim=1, mode='c'] alphas = \
            np.zeros(points.shape[0], dtype=np.float32)
        cdef cnp.ndarray[float, ndim=1, mode='c'] betas = \
            np.zeros(points.shape[0], dtype=np.float32)
        cdef cnp.ndarray[int, ndim=1, mode='c'] indexes = \
            np.zeros(points.shape[0], dtype=np.int32)

        # fill the arrays with the C results
        intersect(self.opcode_model, &points[0, 0], points.shape[0],
                  &indexes[0], &alphas[0], &betas[0])
        return indexes, alphas, betas
