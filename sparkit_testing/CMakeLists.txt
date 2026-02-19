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
