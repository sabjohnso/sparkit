find_package(Catch2 REQUIRED)

macro(sparkit_add_test test_name)
  set(exe_name ${test_name}_test)
  set(src_name ${exe_name}.cpp)
  add_executable(${exe_name} ${src_name})
  target_link_libraries(${exe_name} PRIVATE
    sparkit::header
    sparkit::shared
    Catch2::Catch2WithMain)
  set_target_properties(${exe_name}
    PROPERTIES
    CXX_STANDARD ${sparkit_CXX_STANDARD})
  add_test(${test_name} ${exe_name})
endmacro()

sparkit_add_test(shape shape_test shape_test.cpp)
sparkit_add_test(compressed_data compressed_data_test compressed_data_test.cpp)
sparkit_add_test(sparse_matrix sparse_matrix_test sparse_matrix_test.cpp)
sparkit_add_test(sparse_blas sparse_blas_test sparse_blas_test.cpp)
sparkit_add_test(compressed_column_data compressed_column_data_test compressed_column_data_test.cpp)
sparkit_add_test(compressed_column_matrix compressed_column_matrix_test compressed_column_matrix_test.cpp)
sparkit_add_test(modified_sparse_row_data modified_sparse_row_data_test modified_sparse_row_data_test.cpp)
sparkit_add_test(modified_sparse_row_matrix modified_sparse_row_matrix_test modified_sparse_row_matrix_test.cpp)
sparkit_add_test(diagonal_data diagonal_data_test diagonal_data_test.cpp)
sparkit_add_test(diagonal_matrix diagonal_matrix_test diagonal_matrix_test.cpp)
sparkit_add_test(ellpack_data ellpack_data_test ellpack_data_test.cpp)
sparkit_add_test(ellpack_matrix ellpack_matrix_test ellpack_matrix_test.cpp)
sparkit_add_test(block_sparse_row_data block_sparse_row_data_test block_sparse_row_data_test.cpp)
sparkit_add_test(block_sparse_row_matrix block_sparse_row_matrix_test block_sparse_row_matrix_test.cpp)
sparkit_add_test(jagged_diagonal_data jagged_diagonal_data_test jagged_diagonal_data_test.cpp)
sparkit_add_test(jagged_diagonal_matrix jagged_diagonal_matrix_test jagged_diagonal_matrix_test.cpp)
