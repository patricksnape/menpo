# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from menpo.cy_utils cimport dtype_from_memoryview


# Allow the centres to be of different type from the images
ctypedef fused IMAGE_TYPES:
    double
    float
ctypedef fused CENTRES_TYPES:
    double
    float


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calc_augmented_centers(CENTRES_TYPES[:, :] centres,
                                 Py_ssize_t[:, :] offsets,
                                 Py_ssize_t[:, :] augmented_centers):
    cdef:
        size_t total_index = 0, i = 0, j = 0
        size_t n_centres = centres.shape[0]
        size_t n_offsets = offsets.shape[0]

    for i in range(n_centres):
        for j in range(n_offsets):
            augmented_centers[total_index, 0] = <Py_ssize_t> (centres[i, 0] + offsets[j, 0])
            augmented_centers[total_index, 1] = <Py_ssize_t> (centres[i, 1] + offsets[j, 1])
            total_index += 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calc_slices(Py_ssize_t[:, :] centres,
                      size_t image_shape0,
                      size_t image_shape1,
                      size_t patch_shape0,
                      size_t patch_shape1,
                      size_t half_patch_shape0,
                      size_t half_patch_shape1,
                      size_t add_to_patch0,
                      size_t add_to_patch1,
                      Py_ssize_t[:, :] ext_s_min,
                      Py_ssize_t[:, :] ext_s_max,
                      Py_ssize_t[:, :] ins_s_min,
                      Py_ssize_t[:, :] ins_s_max):
    cdef:
        size_t i = 0, c_min_new0 = 0, c_min_new1 = 0, c_max_new0 = 0, c_max_new1 = 0
        size_t n_centres = centres.shape[0]

    for i in range(n_centres):
        c_min_new0 = centres[i, 0] - half_patch_shape0
        c_min_new1 = centres[i, 1] - half_patch_shape1
        c_max_new0 = centres[i, 0] + half_patch_shape0 + add_to_patch0
        c_max_new1 = centres[i, 1] + half_patch_shape1 + add_to_patch1

        ext_s_min[i, 0] = c_min_new0
        ext_s_min[i, 1] = c_min_new1
        ext_s_max[i, 0] = c_max_new0
        ext_s_max[i, 1] = c_max_new1

        if ext_s_min[i, 0] < 0:
            ext_s_min[i, 0] = 0
        if ext_s_min[i, 1] < 0:
            ext_s_min[i, 1] = 0
        if ext_s_min[i, 0] > image_shape0:
            ext_s_min[i, 0] = image_shape0 - 1
        if ext_s_min[i, 1] > image_shape1:
            ext_s_min[i, 1] = image_shape1 - 1

        if ext_s_max[i, 0] < 0:
            ext_s_max[i, 0] = 0
        if ext_s_max[i, 1] < 0:
            ext_s_max[i, 1] = 0
        if ext_s_max[i, 0] > image_shape0:
            ext_s_max[i, 0] = image_shape0 - 1
        if ext_s_max[i, 1] > image_shape1:
            ext_s_max[i, 1] = image_shape1 - 1

        ins_s_min[i, 0] = ext_s_min[i, 0] - c_min_new0
        ins_s_min[i, 1] = ext_s_min[i, 1] - c_min_new1

        ins_s_max[i, 0] = ext_s_max[i, 0] - c_max_new0 + patch_shape0
        if ins_s_max[i, 0] < 0:
            ins_s_max[i, 0] = 0
        ins_s_max[i, 1] = ext_s_max[i, 1] - c_max_new1 + patch_shape1
        if ins_s_max[i, 1] < 0:
            ins_s_max[i, 1] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void slice_image(IMAGE_TYPES[:, :, :] image,
                      size_t n_channels,
                      size_t n_centres,
                      size_t n_offsets,
                      Py_ssize_t[:, :] ext_s_min,
                      Py_ssize_t[:, :] ext_s_max,
                      Py_ssize_t[:, :] ins_s_min,
                      Py_ssize_t[:, :] ins_s_max,
                      IMAGE_TYPES[:, :, :, :, :] patches):
    cdef size_t total_index = 0, i = 0, j = 0

    for i in range(n_centres):
        for j in range(n_offsets):
            patches[i,
                    j,
                    :,
                    ins_s_min[total_index, 0]:ins_s_max[total_index, 0],
                    ins_s_min[total_index, 1]:ins_s_max[total_index, 1]
            ] = \
            image[:,
                  ext_s_min[total_index, 0]:ext_s_max[total_index, 0],
                  ext_s_min[total_index, 1]:ext_s_max[total_index, 1]]
            total_index += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef extract_patches(IMAGE_TYPES[:, :, :] image,
                      CENTRES_TYPES[:, :] centres,
                      size_t[:] patch_shape,
                      Py_ssize_t[:, :] offsets):
    cdef:
        size_t n_centres = centres.shape[0]
        size_t n_offsets = offsets.shape[0]
        size_t n_augmented_centres = n_centres * n_offsets

        size_t half_patch_shape0 = patch_shape[0] / 2
        size_t half_patch_shape1 = patch_shape[1] / 2
        size_t add_to_patch0 = patch_shape[0] % 2
        size_t add_to_patch1 = patch_shape[1] % 2
        size_t patch_shape0 = patch_shape[0]
        size_t patch_shape1 = patch_shape[1]
        size_t image_shape0 = image.shape[1]
        size_t image_shape1 = image.shape[2]
        size_t n_channels = image.shape[0]

        # Although it is faster to use malloc in this case, the change in syntax
        # and the mental overhead of handling freeing memory is not considered
        # worth it for these buffers. From simple tests it seems you only begin
        # to see a performance difference when you have
        # n_augmented_centres >~ 5000
        Py_ssize_t[:, :] augmented_centers = np.empty([n_augmented_centres, 2], dtype=np.intp)
        Py_ssize_t[:, :] ext_s_max = np.empty([n_augmented_centres, 2], dtype=np.intp)
        Py_ssize_t[:, :] ext_s_min = np.empty([n_augmented_centres, 2], dtype=np.intp)
        Py_ssize_t[:, :] ins_s_max = np.empty([n_augmented_centres, 2], dtype=np.intp)
        Py_ssize_t[:, :] ins_s_min = np.empty([n_augmented_centres, 2], dtype=np.intp)

        output_dtype = dtype_from_memoryview(image)
        IMAGE_TYPES[:, :, :, :, :] patches = np.zeros(
            [n_centres, n_offsets, n_channels, patch_shape0, patch_shape1],
            dtype=output_dtype)

    calc_augmented_centers(centres, offsets, augmented_centers)
    calc_slices(augmented_centers,
                image_shape0,
                image_shape1,
                patch_shape0,
                patch_shape1,
                half_patch_shape0,
                half_patch_shape1,
                add_to_patch0,
                add_to_patch1,
                ext_s_min,
                ext_s_max,
                ins_s_min,
                ins_s_max)
    slice_image(image,
                n_channels,
                n_centres,
                n_offsets,
                ext_s_min,
                ext_s_max,
                ins_s_min,
                ins_s_max,
                patches)

    # No copy (as ndarray)
    return np.asarray(patches)
