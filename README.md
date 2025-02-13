# hamming

Simple library comparing different implementations of computing hamming distance matrices.

## build
'OpenMP' is the only requirement in addition to a standard C++ toolchain.

'''bash
cmake -B build && --build build
'''

## executables
'hamming_test' some unit tests confirming the correctness of the implementations
'hamming_bm' some benchmarks regarding the implementations concering just the hamming distance implementation
'hamming_bm_mat' some benchmarks regarding the implementation of the hamming distance matrix
