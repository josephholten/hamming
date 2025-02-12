#ifndef HAMMING_H
#define HAMMING_H

#include <functional>
#include <vcl/vectorclass.h>
#include <cstdint>

typedef uint8_t Base;

size_t hamming_batch(const size_t batch_size, const Base* x, const Base* y);
size_t hamming_seq_naive(size_t n, const Base* x, const Base* y);
size_t hamming_seq_branchless(size_t n, const Base* x, const Base* y);
size_t hamming_seq_vectorized(size_t n, const Base* x, const Base* y);
size_t hamming_par_vectorized(size_t n, const Base* x, const Base* y);

void hamming_matrix_seq_nonblocked(
    size_t* dist, // NxN matrix
    const Base* data, // N*L array
    size_t N,
    size_t L
);

void hamming_matrix_pargrid_nonblocked(
    size_t* dist, // NxN matrix
    const Base* data, // N*L array
    size_t N,
    size_t L
);

void hamming_matrix_parkernel_nonblocked(
    size_t* dist, // NxN matrix
    const Base* data,
    size_t N,
    size_t L
);

void hamming_matrix_seq_blocked(
    size_t* dist, // NxN matrix
    const Base* data, // NxL array
    size_t N,
    size_t L
);

#endif /* HAMMING_H */

