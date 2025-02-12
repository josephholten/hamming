cmake_minimum_required(VERSION 3.27)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

project(hamming)

include_directories(.)

add_subdirectory(googletest)
add_subdirectory(benchmark)

enable_testing()

#set(OPT_FLAGS )
#set(OPT_FLAGS -mavx2 -mfma)
#set(OPT_FLAGS -O2 -mavx2 -mfma)
set(OPT_FLAGS -Ofast -ffast-math -mavx2 -mfma)

add_library(hamming hamming.cpp)
target_compile_options(hamming PRIVATE ${OPT_FLAGS})


add_executable(hamming_test hamming_test.cpp)
target_link_libraries(hamming_test hamming GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(hamming_test)

add_executable(hamming_bm hamming_bm.cpp)
target_link_libraries(hamming_bm hamming benchmark::benchmark)
target_compile_options(hamming_bm PRIVATE ${OPT_FLAGS})

#add_executable(hamming_mat_bm hamming_mat_bm.cpp ${SOURCES})
#target_include_directories(hamming_mat_bm PRIVATE ${PROJECT_SOURCE_DIR})
#target_link_libraries(hamming_mat_bm OpenMP::OpenMP_CXX)
#target_compile_options(hamming_mat_bm PRIVATE -m64 -O3 -mfma -mavx2 -fsycl)

#add_executable(hamming_vec_bm hamming_vec_bm.cpp ${SOURCES})
#target_link_libraries(hamming_vec_bm OpenMP::OpenMP_CXX)
#target_compile_options(hamming_vec_bm PRIVATE -m64 -O3 -mfma -mavx2 -fsycl)
#target_include_directories(hamming_vec_bm PRIVATE ${PROJECT_SOURCE_DIR})

