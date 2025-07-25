

add_library(seqr
    # FlatIndex.cpp
    # ScoreQueue.cpp
    utils.cpp
    SearchResult.cpp
    # types.cpp
    # RaggedArray.cpp
    FlatSearch.cpp
)

target_include_directories(seqr
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # if headers are in the same folder
        ${HALF_INCLUDE_DIR}
)

target_link_libraries(seqr ${OpenBLAS_LIB} OpenMP::OpenMP_CXX)
