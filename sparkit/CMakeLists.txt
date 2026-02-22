configure_file(config.hpp.in config.hpp)

add_library(sparkit_header INTERFACE)
set_target_properties(sparkit_header
  PROPERTIES
  EXPORT_NAME header)
target_include_directories(sparkit_header
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${sparkit_INSTALL_INCLUDEDIR}>)
set_target_properties(sparkit_header PROPERTIES EXPORT_NAME header)
add_library(sparkit::header ALIAS sparkit_header)

add_library(sparkit_shared_library SHARED
  data/Shape.hpp data/Shape.cpp
  data/Index.hpp data/Index.cpp
  data/Coordinate_sparsity.hpp data/Coordinate_sparsity.cpp
  data/Coordinate_sparsity_Impl.hpp data/Coordinate_sparsity_Impl.cpp
  data/Compressed_row_sparsity.hpp data/Compressed_row_sparsity.cpp
  data/Compressed_row_sparsity_Impl.hpp data/Compressed_row_sparsity_Impl.cpp
  data/Compressed_column_sparsity.hpp data/Compressed_column_sparsity.cpp
  data/Compressed_column_sparsity_Impl.hpp data/Compressed_column_sparsity_Impl.cpp
  data/Modified_sparse_row_sparsity.hpp data/Modified_sparse_row_sparsity.cpp
  data/Modified_sparse_row_sparsity_Impl.hpp data/Modified_sparse_row_sparsity_Impl.cpp
  data/Diagonal_sparsity.hpp data/Diagonal_sparsity.cpp
  data/Diagonal_sparsity_Impl.hpp data/Diagonal_sparsity_Impl.cpp
  data/Ellpack_sparsity.hpp data/Ellpack_sparsity.cpp
  data/Ellpack_sparsity_Impl.hpp data/Ellpack_sparsity_Impl.cpp
  data/Block_sparse_row_sparsity.hpp data/Block_sparse_row_sparsity.cpp
  data/Block_sparse_row_sparsity_Impl.hpp data/Block_sparse_row_sparsity_Impl.cpp
  data/Jagged_diagonal_sparsity.hpp data/Jagged_diagonal_sparsity.cpp
  data/Jagged_diagonal_sparsity_Impl.hpp data/Jagged_diagonal_sparsity_Impl.cpp
  data/Symmetric_compressed_row_sparsity.hpp data/Symmetric_compressed_row_sparsity.cpp
  data/Symmetric_compressed_row_sparsity_Impl.hpp data/Symmetric_compressed_row_sparsity_Impl.cpp
  data/Symmetric_coordinate_sparsity.hpp data/Symmetric_coordinate_sparsity.cpp
  data/Symmetric_coordinate_sparsity_Impl.hpp data/Symmetric_coordinate_sparsity_Impl.cpp
  data/Symmetric_block_sparse_row_sparsity.hpp data/Symmetric_block_sparse_row_sparsity.cpp
  data/Symmetric_block_sparse_row_sparsity_Impl.hpp data/Symmetric_block_sparse_row_sparsity_Impl.cpp
  data/conversions.hpp data/conversions.cpp
  data/matrix_market.hpp data/matrix_market.cpp
  data/harwell_boeing.hpp data/harwell_boeing.cpp
  data/permutation.hpp data/permutation.cpp
  data/reordering.hpp data/reordering.cpp
  data/elimination_tree.hpp data/elimination_tree.cpp
  data/symbolic_cholesky.hpp data/symbolic_cholesky.cpp)

set_target_properties(sparkit_shared_library
  PROPERTIES
    CXX_STANDARD ${sparkit_CXX_STANDARD}
    EXPORT_NAME shared)
target_link_libraries(sparkit_shared_library
  PUBLIC sparkit::header
  PRIVATE nlohmann_json::nlohmann_json)
add_library(sparkit::shared ALIAS sparkit_shared_library)

install(TARGETS sparkit_header sparkit_shared_library
  EXPORT sparkit_EXPORTS
  RUNTIME DESTINATION ${sparkit_INSTALL_BINDIR}
  LIBRARY DESTINATION ${sparkit_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${sparkit_INSTALL_LIBDIR})

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/config.hpp
  DESTINATION ${sparkit_INSTALL_INCLUDEDIR}/sparkit)
