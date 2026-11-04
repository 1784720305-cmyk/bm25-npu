if(BM25_BUILD_BENCHMARKS OR BUILD_TESTING)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()
if(BM25_INDEXSDK_SOURCE)
    set(BM25_INDEXSDK_INCLUDE "${BM25_INDEXSDK_SOURCE}/feature_retrieval/src/ascendfaiss")
    if(NOT EXISTS "${BM25_INDEXSDK_INCLUDE}/common/ErrorCode.h")
        message(FATAL_ERROR "BM25_INDEXSDK_SOURCE must point to an actual IndexSDK source tree")
    endif()
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "Use GCC/Clang with the declared FP32 options")
endif()

# Internal header implementation; this is not yet an installed IndexSDK public API.
add_library(bm25_common INTERFACE)
add_library(AscendBm25::common ALIAS bm25_common)
set_target_properties(bm25_common PROPERTIES EXPORT_NAME common)
target_include_directories(bm25_common INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${BM25_INSTALL_INCLUDEDIR}>")
target_compile_features(bm25_common INTERFACE cxx_std_17)
target_compile_options(bm25_common INTERFACE
    -ffp-contract=off -fno-fast-math
    "$<BUILD_INTERFACE:-O3;-Wall;-Wextra;-fstack-protector-all;-fno-strict-aliasing;-Wreturn-type;-fno-common>")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(bm25_common INTERFACE "$<BUILD_INTERFACE:_FORTIFY_SOURCE=2>")
    target_compile_options(bm25_common INTERFACE "$<BUILD_INTERFACE:-fPIC>")
    target_link_options(bm25_common INTERFACE "$<BUILD_INTERFACE:-Wl,-z,relro;-Wl,-z,now;-Wl,-z,noexecstack>")
endif()

# Preserve command-line executable locations used by the benchmark orchestrator.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(BM25_BUILD_NPU)
    add_subdirectory(ascendhost)
endif()
if(BM25_BUILD_BENCHMARKS)
    add_subdirectory(test/ascendhost/bm25)
endif()
if(BUILD_TESTING)
    add_subdirectory(ut/cases/bm25)
endif()
