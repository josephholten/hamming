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

const size_t L = 600000;

int main(int argc, char** argv) {
    const size_t M = std::stoi(argv[1]);

    const std::vector<std::vector<Base>> sequences = random_seqdata(L,M);
    std::vector<size_t> Dist(M*M, 0);

    auto start = std::chrono::high_resolution_clock::now();
    // hamming_matrix_seq_nonblocked(Dist, sequences);
    // hamming_matrix_parkernel_nonblocked(Dist, sequences);
    // hamming_matrix_pargrid_nonblocked(Dist, sequences);
    // hamming_matrix_seq_blocked(Dist, sequences);
    // hamming_matrix_par_blocked(Dist, sequences);
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);
    size_t chars = M * M / 2 * L;
    std::cout << chars / 1e9 << " Gchars in " << duration << std::endl;
    std::cout << chars / duration.count() / 1e9 << " Gchar/s" << std::endl;
}
