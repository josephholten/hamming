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

typedef uint8_t Base;

#include "seqdata_io.h"
#include "hamming.h"

const size_t L = 32768;

volatile size_t D = 0;

int main(int argc, char** argv) {
    const size_t M = std::stoi(argv[1]);

    // const std::vector<std::vector<Base>> sequences = random_seqdata(L,M);
    const std::vector<Base> x = random_seqdata(L);
    const std::vector<Base> y = random_seqdata(L);
    std::vector<size_t> Dist(M*M, 0);

    auto start = std::chrono::high_resolution_clock::now();
    // hamming_matrix_seq_nonblocked(Dist, sequences);
    // hamming_matrix_parkernel_nonblocked(Dist, sequences);
    // hamming_matrix_pargrid_nonblocked(Dist, sequences);
    // hamming_matrix_seq_blocked(Dist, sequences);
    // hamming_matrix_par_blocked(Dist, sequences);
    // D += hamming_seq_naive(L, x.data(), y.data());
    D += hamming_seq_vectorized(L, x.data(), y.data());
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);
    size_t chars = L;
    std::cout << chars / 1e9 << " Gchars in " << duration << std::endl;
    std::cout << chars / duration.count() / 1e9 << " Gchar/s" << std::endl;
}
