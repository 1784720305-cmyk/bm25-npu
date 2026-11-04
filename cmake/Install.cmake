set(BM25_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/AscendBm25")
set(BM25_HAS_HOST OFF)
set(BM25_HAS_INDEXSDK OFF)
install(TARGETS bm25_common EXPORT AscendBm25CommonTargets COMPONENT Development)
install(EXPORT AscendBm25CommonTargets NAMESPACE AscendBm25::
    DESTINATION "${BM25_CMAKE_INSTALL_DIR}" COMPONENT Development)

install(DIRECTORY "${BM25_SOURCE_ROOT}/feature_retrieval/src/ascendfaiss/common/bm25"
    DESTINATION "${BM25_INSTALL_INCLUDEDIR}/common" COMPONENT Development FILES_MATCHING PATTERN "*.h")

if(TARGET bm25_host)
    set(BM25_HAS_HOST ON)
    install(TARGETS bm25_host EXPORT AscendBm25HostTargets
        ARCHIVE DESTINATION "${BM25_INSTALL_LIBDIR}" COMPONENT Development)
    install(EXPORT AscendBm25HostTargets NAMESPACE AscendBm25::
        DESTINATION "${BM25_CMAKE_INSTALL_DIR}" COMPONENT Development)
endif()
if(TARGET bm25_indexsdk)
    set(BM25_HAS_INDEXSDK ON)
    install(TARGETS bm25_indexsdk EXPORT AscendBm25IndexSDKTargets
        ARCHIVE DESTINATION "${BM25_INSTALL_LIBDIR}" COMPONENT Development)
    install(EXPORT AscendBm25IndexSDKTargets NAMESPACE AscendBm25::
        DESTINATION "${BM25_CMAKE_INSTALL_DIR}" COMPONENT Development)
    install(FILES "${BM25_SOURCE_ROOT}/feature_retrieval/src/ascendfaiss/ascendhost/include/index/AscendIndexBm25.h"
        DESTINATION "${BM25_INSTALL_INCLUDEDIR}/ascendhost/include/index" COMPONENT Development)
endif()

configure_package_config_file("${BM25_SOURCE_ROOT}/cmake/AscendBm25Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/AscendBm25Config.cmake"
    INSTALL_DESTINATION "${BM25_CMAKE_INSTALL_DIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/AscendBm25Config.cmake"
    DESTINATION "${BM25_CMAKE_INSTALL_DIR}" COMPONENT Development)
