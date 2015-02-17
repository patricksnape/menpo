#include <stdio.h>

static inline size_t SUB2IND(const size_t j, const size_t i, const size_t k,
                             const size_t row_size, const size_t col_size,
                             const size_t n_channels) {
    return (i + col_size * j) + (row_size * col_size * k);
}

template<typename T>
void central_difference(const T* in, const size_t rows, const size_t cols,
                        const size_t n_channels,
                        T* out) {
    const size_t n_output_channels = n_channels * 2;

    // This pragma only really helps for very large images with a few channels
    // Therefore, it is much faster for large image feature computation, but
    // will not increase the speed of image fitting.
    #pragma omp parallel for if(n_channels >= 5 && rows >= 1000 && cols >= 1000)
    for (size_t k = 0; k < n_channels; ++k) {
        size_t output_index = 0;
        // row-derivative
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                if (j == 0) {
                    output_index = SUB2IND(0, i, k, rows, cols, n_output_channels);
                    out[output_index] = in[SUB2IND(1, i, k, rows, cols, n_channels)] - in[SUB2IND(0, i, k, rows, cols, n_channels)];
                }
                else if (j == rows - 1) {
                    output_index = SUB2IND(j, i, k, rows, cols, n_output_channels);
                    out[output_index] = in[SUB2IND(j, i, k, rows, cols, n_channels)] - in[SUB2IND(j - 1, i, k, rows, cols, n_channels)];
                }
                else {
                    output_index = SUB2IND(j, i, k, rows, cols, n_output_channels);
                    out[output_index] = (in[SUB2IND(j + 1, i, k, rows, cols, n_channels)] - in[SUB2IND(j - 1, i, k, rows, cols, n_channels)]) / 2.0;
                }
            }
        }

        // column-derivative
        for (size_t j = 0; j < rows; ++j) {
            for (size_t i = 0; i < cols; ++i) {
                if (i == 0) {
                    output_index = SUB2IND(j, 0, n_channels + k , rows, cols, n_output_channels);
                    out[output_index] = in[SUB2IND(j, 1, k, rows, cols, n_channels)] - in[SUB2IND(j, 0, k, rows, cols, n_channels)];
                }
                else if (i == cols - 1) {
                    output_index = SUB2IND(j, cols - 1, n_channels + k, rows, cols, n_output_channels);
                    out[output_index] = in[SUB2IND(j, i, k, rows, cols, n_channels)] - in[SUB2IND(j, i - 1, k, rows, cols, n_channels)];
                }
                else {
                    output_index = SUB2IND(j, i, n_channels + k, rows, cols, n_output_channels);
                    out[output_index] = (in[SUB2IND(j, i + 1, k, rows, cols, n_channels)] - in[SUB2IND(j, i - 1, k, rows, cols, n_channels)]) / 2.0;
                }
            }
        }
    }
}
