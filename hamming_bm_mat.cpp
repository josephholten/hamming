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

#include <seqdata_io.h>
#include <hamming.h>

// should be chosen such that it is larger than cache
size_t L = 1<<15; // approx 32k
size_t N = 1<<10;
std::vector<Base> x = random_seqdata(N*L);
std::vector<size_t> dist(N*N,0);

void setBasesProcessed(benchmark::State& state) {
  using benchmark::Counter;
  state.counters["bases_per_second"] =
    Counter(static_cast<double>(state.range(0)*state.range(0)*L),
        Counter::kIsIterationInvariantRate, Counter::kIs1024);
  state.counters["bytes"] =
    Counter(static_cast<double>(state.range(0)*L),
        Counter::kDefaults, Counter::kIs1024);
}

static void BM_hamming_matrix_seq_nonblocked(benchmark::State& state) {
    for (auto _ : state) {
      hamming_matrix_seq_nonblocked(dist.data(),x.data(),state.range(0),L);
    }
    setBasesProcessed(state);
}

BENCHMARK(BM_hamming_matrix_seq_nonblocked)->Arg(1<<4)->Arg(1<<8)->Arg(1<<9)->Arg(1<<10);

static void BM_hamming_matrix_seq_blocked(benchmark::State& state) {
    for (auto _ : state) {
      hamming_matrix_seq_blocked(dist.data(),x.data(),state.range(0),L);
    }
    setBasesProcessed(state);
}

BENCHMARK(BM_hamming_matrix_seq_blocked)->Arg(1<<4)->Arg(1<<8)->Arg(1<<9)->Arg(1<<10);





BENCHMARK_MAIN();
