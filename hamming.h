#include <functional>

size_t hamming_seq_naive(size_t n, const Base* x, const Base* y) {
    size_t dist = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] != y[i])
            dist++;
    }
    return dist;
}

size_t hamming_seq_branchless(size_t n, const Base* x, const Base* y) {
    size_t dist = 0;
    for (size_t i = 0; i < n; i++) {
        dist += x[i] != y[i];
    }
    return dist;
}

size_t hamming_batch(const size_t batch_size, const Base* x, const Base* y) {
    Vec32uc X, Y, Dist = 0; // vector of 32 uint8_t
    Vec32cb Cmp;        // vector of 32 booleans compatible with uint8_t

    size_t vector_size = X.size();

    // we can at most process 255 vectors in a batch as more results in uncontrolled overflow
    assert(batch_size < 255);
    for (size_t i = 0; i < batch_size; i++) {
        // calculate position in memory
        size_t offset = i*vector_size;

        X.load(x + offset);
        Y.load(y + offset);

        Cmp = X != Y;
        // Vec32uc(Vec32cb(true, false, ...)) == Vec32uc(FF, 0, ...)
        Dist += Vec32uc(Cmp); 
    }

    // need to flip: k*FF -> k
    Dist = -Dist;
    return horizontal_add_x(Dist);
}

size_t hamming_seq_vectorized(size_t n, const Base* x, const Base* y) {
    uint64_t dist = 0; // total distance
    
    constexpr size_t vector_size = 32;
    const size_t total_vectors = n / vector_size; // rounds down

    constexpr size_t batch_size = 252; // must be less than 255, and divisible by 4
    const size_t total_batches = total_vectors / batch_size; 

    const size_t left_over_vectors = total_vectors % batch_size;
    const size_t left_over_single  = n % vector_size;

    size_t offset = 0;
    for (size_t batch = 0; batch < total_batches; batch += 1) {
        offset = batch*batch_size*vector_size;
        dist += hamming_batch(batch_size, x + offset, y + offset);
    }

    // need to process the left over vectors
    offset = total_batches*batch_size*vector_size;
    dist += hamming_batch(left_over_vectors, x + offset, y + offset);

    // need to process left over singles
    offset = total_vectors*vector_size;
    dist += hamming_seq_branchless(left_over_single, x + offset, y + offset);

    return dist;
}

size_t hamming_par_vectorized(size_t n, const Base* x, const Base* y) {
    uint64_t dist = 0; // total distance
    
    constexpr size_t vector_size = 32;
    const size_t total_vectors = n / vector_size; // rounds down
    
    constexpr size_t batch_size = 252; // must be less than 255, and divisible by 4
    const size_t total_batches = total_vectors / batch_size; 

    const size_t left_over_vectors = total_vectors % batch_size;
    const size_t left_over_single  = n % vector_size;


    #pragma omp parallel reduction(+: dist) 
    {
        #pragma omp for schedule(static) 
        for (size_t batch = 0; batch < total_batches; batch += 1) {
            size_t offset = batch*batch_size*vector_size;
            dist += hamming_batch(batch_size, x + offset, y + offset);
        }
    }

    // need to process the left over vectors
    size_t offset = total_batches*batch_size*vector_size;
    dist += hamming_batch(left_over_vectors, x + offset, y + offset);

    // need to process left over singles
    offset = total_vectors*vector_size;
    dist += hamming_seq_branchless(left_over_single, x + offset, y + offset);

    return dist;
}

// In the following: assume all sequence data is contiguous in memory
// all vectorized

void hamming_matrix_seq_nonblocked(
    size_t* dist, // NxN matrix
    const Base* data, // N*L array
    size_t N,
    size_t L
) {
    for (size_t row = 0; row < N; row++) {
        for (size_t col = row + 1; col < N; col++) {
            dist[row*N + col] = hamming_seq_vectorized(L, data+row*L, data+col*L);
        }
    }
}

void hamming_matrix_pargrid_nonblocked(
    size_t* dist, // NxN matrix
    const Base* data, // N*L array
    size_t N,
    size_t L
) {
    #pragma omp parallel for schedule(static, 1)
    for (size_t row = 0; row < N; row++) {
        for (size_t col = row + 1; col < N; col++) {
            dist[row*N + col] = hamming_seq_vectorized(L, data+row*L, data+col*L);
        }
    }
}

void hamming_matrix_parkernel_nonblocked(
    size_t* dist, // NxN matrix
    const Base* data,
    size_t N,
    size_t L
) {
    for (size_t row = 0; row < N; row++) {
        for (size_t col = row + 1; col < N; col++) {
            dist[row*N + col] = hamming_par_vectorized(L, data+row*L, data+col*L);
        }
    }
}

void batched_array(const size_t N, const size_t batch_size, const size_t vector_size, std::function<void(size_t, size_t)> f_batch, std::function<void(size_t, size_t)> f_single) {
    size_t offset = 0;
    for (; offset < N; offset += batch_size*vector_size) {
        f_batch(batch_size, offset);
    }

    size_t vectors_remaining = (N-offset)/vector_size;
    f_batch(vectors_remaining, offset);
    offset += vectors_remaining*vector_size;

    size_t singles = N - offset;
    f_single(singles, offset);
}

void blocked_matrix_symm(const size_t N, const size_t B, std::function<void(size_t, size_t)> f) {
    for (size_t row = 0; row < N; row += B) {
        // diagonal block
        size_t col = row;
        for (size_t r = row; r < row + B; r++) {
            for (size_t c = r+1; c < col + B; c++) { // diagonal is zero, so ignore it
                f(r, c);
            }
        }

        // off diagonal blocks
        for (size_t col = row + B; col < N; col += B) { // only need upper triangular
            // this should now all be in cache
            for (size_t r = row; r < row + B; r++) {
                for (size_t c = col; c < col + B; c++) {
                    f(r, c);
                }
            }
        }
    }
}

void hamming_matrix_seq_blocked(
    size_t* dist, // NxN matrix
    const Base* data, // NxL array
    size_t N,
    size_t L
) {
    // reset dist
    std::fill(dist, dist + N*N, 0);

    const size_t number_of_sequences = N;
    const size_t sequence_length = L;

    // compute BxB interactions in block -> 2B loads for BxB computations -> intensity

    // can at most do 255 Vec32uc additions at once
    // but also want 2*block_size*batch_size*vector_size bytes to fit into cache
    // on my machine this should be around 2*4*128*32 bytes = 32 KiB = L1 cache per core

    constexpr size_t block_size = 4;
    constexpr size_t vector_size = 32; // in bytes
    constexpr size_t batch_size = 128;

    if (N % block_size != 0) {
        std::cerr << "ERROR: number of sequences " << N << " must be divisible by the block size " << block_size << std::endl;
    }

    const size_t total_vectors = L / vector_size; // rounds down

    const size_t total_batches = total_vectors / batch_size;

    const size_t left_over_vectors = total_vectors % batch_size;
    const size_t left_over_single  = L % vector_size;

    /*
    std::cout << "L " << L << std::endl;
    std::cout << "batches " << total_batches << " -> " << total_batches*batch_size*vector_size << " R " << left_over_vectors << std::endl;
    std::cout << "vecs " << total_vectors << " -> " << total_vectors*vector_size << " R " << left_over_single << std::endl;
    */

    size_t offset = 0;
    for (size_t batch = 0; batch < total_batches; batch += 1) {
        offset = batch*batch_size*vector_size;
        for (size_t row = 0; row < N; row += block_size) {
            // diagonal block
            size_t col = row;
            for (size_t r = row; r < row + block_size; r++) {
                for (size_t c = r+1; c < col + block_size; c++) { // diagonal is zero, so ignore it
                    dist[r*N+c] += hamming_batch(batch_size, data + r*L + offset, data + c*L + offset);
                }
            }

            // off diagonal blocks
            for (size_t col = row + block_size; col < N; col += block_size) { // only need upper triangular
                // this should now all be in cache
                for (size_t r = row; r < row + block_size; r++) {
                    for (size_t c = col; c < col + block_size; c++) {
                        dist[r*N+c] += hamming_batch(batch_size, data + r*L + offset, data + c*L + offset);
                    }
                }
            }
        }
    }

    // need to process the left over vectors
    offset = total_batches*batch_size*vector_size;
    for (size_t row = 0; row < N; row += block_size) {
        // diagonal block
        size_t col = row;
        for (size_t r = row; r < row + block_size; r++) {
            for (size_t c = r+1; c < col + block_size; c++) { // diagonal is zero, so ignore it
                dist[r*N+c] += hamming_batch(left_over_vectors, data + r*L + offset, data + c*L + offset);
            }
        }

        // off diagonal blocks
        for (size_t col = row + block_size; col < N; col += block_size) { // only need upper triangular
            // this should now all be in cache
            for (size_t r = row; r < row + block_size; r++) {
                for (size_t c = col; c < col + block_size; c++) {
                    dist[r*N+c] += hamming_batch(left_over_vectors, data + r*L + offset, data + c*L + offset);
                }
            }
        }
    }

    // need to process left over singles
    offset = total_vectors*vector_size;
    for (size_t row = 0; row < N; row += block_size) {
        // diagonal block
        size_t col = row;
        for (size_t r = row; r < row + block_size; r++) {
            for (size_t c = r+1; c < col + block_size; c++) { // diagonal is zero, so ignore it
                dist[r*N+c] += hamming_seq_naive(left_over_single, data + r*L + offset, data + c*L + offset);
            }
        }

        // off diagonal blocks
        for (size_t col = row + block_size; col < N; col += block_size) { // only need upper triangular
            // this should now all be in cache
            for (size_t r = row; r < row + block_size; r++) {
                for (size_t c = col; c < col + block_size; c++) {
                    dist[r*N+c] += hamming_seq_naive(left_over_single, data + r*L + offset, data + c*L + offset);
                }
            }
        }
    }


}
