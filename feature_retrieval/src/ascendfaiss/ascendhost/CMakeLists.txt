if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "NPU targets require A5 Linux; CPU-only checks use -DBM25_BUILD_NPU=OFF")
endif()
find_path(ACL_INCLUDE acl/acl.h PATHS "${TOOLKIT_HOME}/include" NO_DEFAULT_PATH)
find_path(META_INCLUDE aclnn/acl_meta.h PATHS "${TOOLKIT_HOME}/include" "${TOOLKIT_HOME}/include/aclnnop" NO_DEFAULT_PATH)
find_library(ACL_LIBRARY ascendcl PATHS "${TOOLKIT_HOME}/lib64" NO_DEFAULT_PATH)
find_library(NNOP_LIBRARY nnopbase PATHS "${TOOLKIT_HOME}/lib64" NO_DEFAULT_PATH)
if(NOT ACL_INCLUDE OR NOT META_INCLUDE OR NOT ACL_LIBRARY OR NOT NNOP_LIBRARY OR NOT BM25_VENDOR_DIR)
    message(FATAL_ERROR "Actual CANN and final runtime aliases required")
endif()
foreach(pair bm25_fused_score_topk/aclnn_bm25_fused_score_top_k.h bm25_topk_merge/aclnn_bm25_top_k_merge.h)
    string(REPLACE "/" "/op_api/include/" header "${pair}")
    if(NOT EXISTS "${BM25_VENDOR_DIR}/${header}")
        message(FATAL_ERROR "Actual generated API missing: ${header}")
    endif()
    get_filename_component(parent "${BM25_VENDOR_DIR}/${header}" DIRECTORY)
    list(APPEND API_INCLUDES "${parent}")
endforeach()
add_library(bm25_host STATIC
    src/bm25/runtime.cpp
    src/bm25/fused_runtime.cpp
    src/bm25/planned_search.cpp
    src/bm25/large_search.cpp)
add_library(AscendBm25::host ALIAS bm25_host)
set_target_properties(bm25_host PROPERTIES EXPORT_NAME host)
target_compile_definitions(bm25_host PUBLIC _GNU_SOURCE)
target_include_directories(bm25_host PUBLIC
    "$<BUILD_INTERFACE:${ACL_INCLUDE}>" "$<BUILD_INTERFACE:${META_INCLUDE}>"
    "$<BUILD_INTERFACE:${API_INCLUDES}>")
target_link_libraries(bm25_host PUBLIC bm25_common
    "$<BUILD_INTERFACE:${NNOP_LIBRARY}>" "$<BUILD_INTERFACE:${ACL_LIBRARY}>"
    "$<INSTALL_INTERFACE:AscendBm25::CANN>" ${CMAKE_DL_LIBS})

if(BM25_INDEXSDK_SOURCE)
    add_library(bm25_indexsdk STATIC src/bm25/AscendIndexBm25.cpp)
    add_library(AscendBm25::indexsdk ALIAS bm25_indexsdk)
    set_target_properties(bm25_indexsdk PROPERTIES EXPORT_NAME indexsdk)
    find_package(Threads REQUIRED)
    target_include_directories(bm25_indexsdk PUBLIC "$<BUILD_INTERFACE:${BM25_INDEXSDK_INCLUDE}>")
    target_link_libraries(bm25_indexsdk PUBLIC bm25_host Threads::Threads
        "$<INSTALL_INTERFACE:AscendBm25::IndexSDKHeaders>")
endif()
