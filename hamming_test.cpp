#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <cinttypes>
#include <stdio.h>
#include <fstream>
#include <array>
#include <vcl/vectorclass.h>
#include <cassert>
#include <chrono>
#include <set>

typedef uint8_t Base;

#include "seqdata_io.h"
#include "hamming.h"

#include <gtest/gtest.h>

void print_vec(const std::vector<Base>& v) {
    for (size_t i = 0; i < v.size(); i++) {
        printf("%c ", v[i]);
    }
    printf("\n");
}

TEST(hamming_seq_naive, human) {
    const std::vector<Base> a = {'A', 'B', 'C', 'D'};
    const std::vector<Base> b = {'A', 'B', 'C', 'C'};
    const std::vector<Base> c = {'D', 'C', 'C', 'C'};

    ASSERT_EQ(a.size(), b.size());
    ASSERT_EQ(b.size(), c.size());

    size_t n = a.size();
    EXPECT_EQ(0, hamming_seq_naive(0, a.data(), c.data()));
    EXPECT_EQ(0, hamming_seq_naive(n, a.data(), a.data()));
    EXPECT_EQ(1, hamming_seq_naive(n, a.data(), b.data()));
    EXPECT_EQ(1, hamming_seq_naive(n, b.data(), a.data()));
    EXPECT_EQ(2, hamming_seq_naive(n, b.data(), c.data()));
    EXPECT_EQ(2, hamming_seq_naive(n, c.data(), b.data()));
    EXPECT_EQ(3, hamming_seq_naive(n, c.data(), a.data()));
    EXPECT_EQ(3, hamming_seq_naive(n, a.data(), c.data()));
}

TEST(hamming_seq_naive, random_reflexive) {
    constexpr size_t L = 1000;
    const std::vector<Base> x = random_seqdata(L);

    for (size_t l = 0; l < L; l++)
        EXPECT_EQ(0, hamming_seq_naive(l, x.data(), x.data())) << "hamming_dist should be reflexive";
}

TEST(hamming_seq_naive, random_distribution) {
    constexpr size_t L = 2<<19;

    std::random_device r;
    std::default_random_engine engine(r());
    std::uniform_int_distribution<size_t> dist(L >> 1, L << 1);

    constexpr double eps = 1e-2;
    constexpr size_t reps = 20;

    for (size_t i = 0; i < reps; i++) {
        const size_t l = dist(engine);

        const std::vector<Base> x = random_seqdata(l);
        const std::vector<Base> y = random_seqdata(l);

        const double avg = (double) hamming_seq_naive(l, x.data(), y.data()) / l;

        EXPECT_NEAR(avg, 0.75, eps);
    }
}

TEST(hamming_seq_naive, random_length) {
    constexpr size_t L = 2<<10;

    std::random_device r;
    std::default_random_engine engine(r());
    std::uniform_int_distribution<size_t> length_dist(L >> 1, L << 1);

    constexpr size_t reps = 10;

    for (size_t i = 0; i < reps; i++) {
        const size_t l = length_dist(engine);
        std::uniform_int_distribution<size_t> index_dist(0, l-1);

        // change M indices
        const size_t M = index_dist(engine);
        std::set<size_t> indices;
        for (size_t m = 0; m < M; m++) {
            size_t idx = -1;
            do {
                idx = index_dist(engine);
            } while(indices.find(idx) != indices.end());
            indices.insert(idx);
        }
        
        ASSERT_EQ(indices.size(), M);

        const std::vector<Base> x = random_seqdata(l);
        std::vector<Base> y = x;

        for (size_t idx : indices) {
            ASSERT_TRUE(idx < l);
            y[idx] = 'X';
        }

        EXPECT_EQ(hamming_seq_naive(l, x.data(), y.data()), M);
    }
}

TEST(hamming_seq_branchless, against_naive) {
    constexpr size_t L = 2<<10;

    std::random_device r;
    std::default_random_engine engine(r());
    std::uniform_int_distribution<size_t> length_dist(L >> 1, L << 1);

    constexpr size_t reps = 10;

    for (size_t i = 0; i < reps; i++) {
        const size_t l = length_dist(engine);
        const std::vector<Base> x = random_seqdata(l);
        const std::vector<Base> y = random_seqdata(l);

        EXPECT_EQ(
            hamming_seq_naive(l, x.data(), y.data()), 
            hamming_seq_branchless(l, x.data(), y.data())
        );
    }
}

TEST(hamming_seq_vectorized, reflexive) {
    constexpr size_t L = 2<<10;

    const std::vector<Base> x = random_seqdata(L);

    EXPECT_EQ(
        0,
        hamming_seq_vectorized(L, x.data(), x.data())
    );
}

TEST(hamming_seq_vectorized, human1) {
    constexpr size_t L = 10000;

    const std::vector<Base> x = random_seqdata(L);
    std::vector<Base> y = x;
    y[0] = 'X';

    EXPECT_EQ(
        1,
        hamming_seq_branchless(L, x.data(), y.data())
    );

    EXPECT_EQ(
        hamming_seq_branchless(L, x.data(), y.data()),
        hamming_seq_vectorized(L, x.data(), y.data())
    );
}

TEST(hamming_seq_vectorized, hamming_batch) {
    std::random_device r;
    std::default_random_engine engine(r());
    std::uniform_int_distribution<size_t> length_dist(4, 252);

    constexpr size_t reps = 100;

    for (size_t i = 0; i < reps; i++) {
        const size_t batch_size = length_dist(engine);
        const size_t l = batch_size * 32;
        const std::vector<Base> x = random_seqdata(l);
        const std::vector<Base> y = random_seqdata(l);
        
        EXPECT_EQ(
            hamming_seq_branchless(l, x.data(), y.data()),
            hamming_batch(batch_size, x.data(), y.data())
        );
    }
}

TEST(hamming_seq_vectorized, human2) {
    const std::vector<size_t> ls = {32, 32*252, 32*258, 32 + 1};
    for (const size_t l : ls) {
        const std::vector<Base> x = random_seqdata(l);
        const std::vector<Base> y = random_seqdata(l);

        EXPECT_EQ(
            hamming_seq_branchless(l, x.data(), y.data()),
            hamming_seq_vectorized(l, x.data(), y.data())
        ) << "l == " << l << " (l % 32) == " << l%32;
    }
}

TEST(hamming_seq_vectorized, against_branchless) {
    constexpr size_t L = 2<<10;

    std::random_device r;
    std::default_random_engine engine(r());
    std::uniform_int_distribution<size_t> length_dist(L >> 1, L << 1);

    constexpr size_t reps = 10;

    for (size_t i = 0; i < reps; i++) {
        const size_t l = length_dist(engine);
        const std::vector<Base> x = random_seqdata(l);
        const std::vector<Base> y = random_seqdata(l);

        EXPECT_EQ(
            hamming_seq_branchless(l, x.data(), y.data()),
            hamming_seq_vectorized(l, x.data(), y.data())
        );
    }
}

TEST(hamming_par_vectorized, against_branchless) {
    constexpr size_t L = 2<<10;

    std::random_device r;
    std::default_random_engine engine(r());
    std::uniform_int_distribution<size_t> length_dist(L >> 1, L << 1);

    constexpr size_t reps = 10;

    for (size_t i = 0; i < reps; i++) {
        const size_t l = length_dist(engine);
        const std::vector<Base> x = random_seqdata(l);
        const std::vector<Base> y = random_seqdata(l);

        EXPECT_EQ(
            hamming_seq_branchless(l, x.data(), y.data()),
            hamming_par_vectorized(l, x.data(), y.data())
        );
    }
}

TEST(hamming_matrix_seq_nonblocked, human) {
    const size_t N = 3;
    const size_t L = 4;

    const std::vector<Base> data = {
        'A', 'B', 'C', 'D', // a
        'A', 'B', 'C', 'C', // b
        'D', 'C', 'C', 'C', // c
    };

    ASSERT_EQ(data.size(), N*L);
    
    std::vector<size_t> dist(N*N, 0);

    // only lower triangular
    const std::vector<size_t> dist_true = {
        0, 1, 3, // a -> a, b, c
        0, 0, 2, // b -> a, b, c
        0, 0, 0, // c -> a, b, c
    };

    ASSERT_EQ(dist_true.size(), dist.size());

    hamming_matrix_seq_nonblocked(dist.data(), data.data(), N, L);

    for (size_t k = 0; k < N*N; k++)
        EXPECT_EQ(dist[k], dist_true[k]);
}

TEST(hamming_matrix_pargrid, against_seq_nonblocked) {
    constexpr size_t L = 32768;
    constexpr size_t N = 16;
    const std::vector<Base> seq = random_seqdata(L*N);
    std::vector<size_t> dist(N*N, 0);
    std::vector<size_t> dist_true(N*N, 0);

    hamming_matrix_seq_nonblocked(dist_true.data(), seq.data(), N, L);
    hamming_matrix_pargrid_nonblocked(dist.data(), seq.data(), N, L);

    for (size_t k = 0; k < N*N; k++)
        EXPECT_EQ(dist[k], dist_true[k]);
}

TEST(hamming_matrix_seq_blocked, against_seq_nonblocked_single_batch) {
    constexpr size_t L = 128*32;
    constexpr size_t N = 4;
    const std::vector<Base> seq = random_seqdata(L*N);
    std::vector<size_t> dist(N*N, 0);
    std::vector<size_t> dist_true(N*N, 0);

    hamming_matrix_seq_blocked(dist.data(), seq.data(), N, L);
    hamming_matrix_seq_nonblocked(dist_true.data(), seq.data(), N, L);

    for (size_t k = 0; k < N*N; k++)
        EXPECT_EQ(dist[k], dist_true[k]);
}

TEST(hamming_matrix_seq_blocked, against_seq_nonblocked_multi_batch) {
    constexpr size_t L = 128*32*2;
    constexpr size_t N = 4;
    const std::vector<Base> seq = random_seqdata(L*N);
    std::vector<size_t> dist(N*N, 0);
    std::vector<size_t> dist_true(N*N, 0);

    hamming_matrix_seq_blocked(dist.data(), seq.data(), N, L);
    hamming_matrix_seq_nonblocked(dist_true.data(), seq.data(), N, L);

    for (size_t k = 0; k < N*N; k++)
        EXPECT_EQ(dist[k], dist_true[k]);
}

TEST(hamming_matrix_seq_blocked, against_seq_nonblocked_random_length) {
    constexpr size_t N = 32;
    constexpr size_t L = 1<<14;

    std::random_device r;
    std::default_random_engine engine(r());
    std::uniform_int_distribution<size_t> length_dist(L >> 1, L << 1);

    std::vector<size_t> dist(N*N, 0);
    std::vector<size_t> dist_true(N*N, 0);

    constexpr size_t iterations = 10;

    for (size_t i = 0; i < iterations; i++) {
        const size_t l = length_dist(engine);

        const std::vector<Base> seq = random_seqdata(l*N);

        hamming_matrix_seq_blocked(dist.data(), seq.data(), N, l);
        hamming_matrix_seq_nonblocked(dist_true.data(), seq.data(), N, l);

        for (size_t k = 0; k < N*N; k++)
            EXPECT_EQ(dist[k], dist_true[k]);
    }
}
