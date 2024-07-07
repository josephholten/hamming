#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <cinttypes>
#include <stdio.h>
#include <fstream>
#include <array>
#include <vcl/vectorclass.h>
#include <benchmark/benchmark.h>

typedef uint8_t Base;
volatile size_t D = 0;

#include "seqdata_io.h"
#include "hamming.h"

// should be chosen such that it is larger than cache
size_t N = 8<<24;
std::vector<Base> x = random_seqdata(N);
std::vector<Base> y = random_seqdata(N);

static void BM_hamming_seq_naive(benchmark::State& state) {
    for (auto _ : state) {
        D += hamming_seq_naive(state.range(0), x.data(), y.data());
    }
}

// BENCHMARK(BM_hamming_seq_naive)->Range(8, N);

static void BM_hamming_seq_branchless(benchmark::State& state) {
    for (auto _ : state) {
        D += hamming_seq_branchless(state.range(0), x.data(), y.data());
    }
}

// BENCHMARK(BM_hamming_seq_branchless)->Range(8, N);

static void BM_hamming_seq_vectorized(benchmark::State& state) {
    for (auto _ : state) {
        D += hamming_seq_vectorized(state.range(0), x.data(), y.data());
    }
}

// BENCHMARK(BM_hamming_seq_vectorized)->Range(8, N);

static void BM_hamming_par_vectorized(benchmark::State& state) {
    for (auto _ : state) {
        D += hamming_par_vectorized(state.range(0), x.data(), y.data());
    }
}

// BENCHMARK(BM_hamming_par_vectorized)->Range(8, N);

BENCHMARK_MAIN();
