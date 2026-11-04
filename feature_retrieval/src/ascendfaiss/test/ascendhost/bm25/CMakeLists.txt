set(CORE_SOURCES
    cmake/Install.cmake
    cmake/AscendBm25Config.cmake.in
    feature_retrieval/src/ascendfaiss/common/bm25/topk_heap.h
    feature_retrieval/src/ascendfaiss/common/bm25/error_info.h
    feature_retrieval/src/ascendfaiss/ascendhost/include/impl/Bm25RuntimeBinding.h
    feature_retrieval/src/ascendfaiss/ascendhost/src/bm25/runtime.h
    feature_retrieval/src/ascendfaiss/ascendhost/src/bm25/runtime.cpp

    feature_retrieval/src/ascendfaiss/common/bm25/error.h
    feature_retrieval/src/ascendfaiss/ascendhost/CMakeLists.txt
    feature_retrieval/src/ascendfaiss/common/bm25/index_data.h
    feature_retrieval/src/ascendfaiss/common/bm25/validation.h
    feature_retrieval/src/ascendfaiss/ascendhost/src/bm25/fused_runtime.cpp
    feature_retrieval/src/ascendfaiss/ascendhost/src/bm25/planned_search.cpp
    feature_retrieval/src/ascendfaiss/ascendhost/src/bm25/large_search.cpp
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/input_io.h
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/search_diagnostic.cpp
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/search_diagnostic.h
    feature_retrieval/src/ascendfaiss/common/bm25/large_search_plan.h feature_retrieval/src/ascendfaiss/common/bm25/fixed_query_packet.h feature_retrieval/src/ascendfaiss/common/bm25/shard_directory.h
    feature_retrieval/src/ascendfaiss/common/bm25/topk_document_shards.h feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/reference/cpu_index_search.h
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/reference/cpu_directory_search.h
    feature_retrieval/src/ascendfaiss/common/bm25/search_inputs.h
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/benchmark_common.h
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/cpu_benchmark.cpp
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/cpu_directory_benchmark.cpp
    feature_retrieval/src/ascendfaiss/ascendhost/src/bm25/fused_runtime.h feature_retrieval/src/ascendfaiss/ascendhost/src/bm25/fused_api.h
    feature_retrieval/src/ascendfaiss/ascendhost/src/bm25/planned_search.h
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/benchmark.cpp feature_retrieval/src/ascendfaiss/ascendhost/src/bm25/large_search.h
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/merge_check.h CMakeLists.txt
    feature_retrieval/src/ascendfaiss/CMakeLists.txt
    feature_retrieval/src/ascendfaiss/test/ascendhost/bm25/CMakeLists.txt
    feature_retrieval/src/ascendfaiss/ut/cases/bm25/CMakeLists.txt)
set(RECORD "#pragma once\n#include <cstddef>\nstatic constexpr const char *BM25_BENCH_SOURCES[][2] = {\n")
foreach(source IN LISTS CORE_SOURCES)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${BM25_SOURCE_ROOT}/${source}")
    file(SHA256 "${BM25_SOURCE_ROOT}/${source}" sha)
    string(APPEND RECORD "{\"${source}\",\"${sha}\"},\n")
endforeach()
string(APPEND RECORD "};\nstatic constexpr std::size_t BM25_BENCH_SOURCE_COUNT=sizeof(BM25_BENCH_SOURCES)/sizeof(BM25_BENCH_SOURCES[0]);\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compiled_benchmark_sources.h" "${RECORD}")

add_executable(bm25_cpu_search_benchmark cpu_benchmark.cpp)
add_executable(bm25_directory_cpu_benchmark cpu_directory_benchmark.cpp)
foreach(target bm25_cpu_search_benchmark bm25_directory_cpu_benchmark)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(${target} PRIVATE bm25_common OpenMP::OpenMP_CXX)
    target_compile_options(${target} PRIVATE -march=native)
endforeach()
if(BM25_BUILD_NPU)
    add_executable(bm25_large_search_benchmark benchmark.cpp search_diagnostic.cpp)
    add_executable(bm25_fused_simd_selector_check selector_check.cpp)
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/selector_check.cpp" CHECK_SHA)
    target_compile_definitions(bm25_fused_simd_selector_check PRIVATE BM25_SIMD_CHECK_SHA="${CHECK_SHA}")
    foreach(target bm25_large_search_benchmark bm25_fused_simd_selector_check)
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_link_libraries(${target} PRIVATE bm25_host)
        target_compile_options(${target} PRIVATE -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra)
        set_target_properties(${target} PROPERTIES BUILD_RPATH "${TOOLKIT_HOME}/lib64")
    endforeach()
    if(BM25_INDEXSDK_SOURCE AND BM25_INDEXSDK_LIBRARY)
        if(NOT EXISTS "${BM25_INDEXSDK_LIBRARY}")
            message(FATAL_ERROR "BM25_INDEXSDK_LIBRARY must name the actual IndexSDK host library")
        endif()
        if(NOT BM25_SECUREC_INCLUDE_DIR)
            find_path(BM25_SDK_SECUREC_HEADER securec.h HINTS "${TOOLKIT_HOME}/include")
        else()
            set(BM25_SDK_SECUREC_HEADER "${BM25_SECUREC_INCLUDE_DIR}")
        endif()
        if(NOT EXISTS "${BM25_SDK_SECUREC_HEADER}/securec.h")
            message(FATAL_ERROR "securec.h required by SDK resource headers; set BM25_SECUREC_INCLUDE_DIR")
        endif()
        add_executable(bm25_sdk_shared_check sdk_shared_check.cpp)
        target_include_directories(bm25_sdk_shared_check PRIVATE "${BM25_INDEXSDK_INCLUDE}/common"
            "${BM25_SDK_SECUREC_HEADER}")
        target_link_libraries(bm25_sdk_shared_check PRIVATE bm25_indexsdk "${BM25_INDEXSDK_LIBRARY}")
        get_filename_component(BM25_SDK_LIBRARY_DIR "${BM25_INDEXSDK_LIBRARY}" DIRECTORY)
        set_target_properties(bm25_sdk_shared_check PROPERTIES BUILD_RPATH "${TOOLKIT_HOME}/lib64;${BM25_SDK_LIBRARY_DIR}")
    endif()
endif()
