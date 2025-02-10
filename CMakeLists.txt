cmake_minimum_required(VERSION 3.27)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

project(hamming)

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest CONFIG REQUIRED)

add_library(hamming hamming.cpp)
target_compile_definitions(hamming PRIVATE -fsycl -march=native)

#set(SOURCES vcl/vectorclass.h hamming.h seqdata_io.h)

#add_executable(hamming_bm hamming_bm.cpp ${SOURCES})
#target_include_directories(hamming_bm PRIVATE ${PROJECT_SOURCE_DIR})

#target_link_libraries(hamming_bm benchmark::benchmark OpenMP::OpenMP_CXX)
#target_compile_options(hamming_bm PRIVATE -m64 -O3 -mfma -mavx2 -fsycl)

#add_executable(hamming_mat_bm hamming_mat_bm.cpp ${SOURCES})
#target_include_directories(hamming_mat_bm PRIVATE ${PROJECT_SOURCE_DIR})

#target_link_libraries(hamming_mat_bm OpenMP::OpenMP_CXX)
#target_compile_options(hamming_mat_bm PRIVATE -m64 -O3 -mfma -mavx2 -fsycl)

#add_executable(hamming_vec_bm hamming_vec_bm.cpp ${SOURCES})
#target_link_libraries(hamming_vec_bm OpenMP::OpenMP_CXX)
#target_compile_options(hamming_vec_bm PRIVATE -m64 -O3 -mfma -mavx2 -fsycl)
#target_include_directories(hamming_vec_bm PRIVATE ${PROJECT_SOURCE_DIR})

enable_testing()

add_executable(hamming_test hamming_test.cpp)
target_link_libraries(hamming_test hamming)

include(GoogleTest)
gtest_discover_tests(hamming_test)
