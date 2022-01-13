find_package(GTest)

macro(sparkit_add_test test_name exe_name)
  add_executable(${exe_name} ${ARGN})
  target_link_libraries(${exe_name} PRIVATE
    sparkit::header
    GTest::gtest_main)
  add_test(${test_name} ${exe_name})
endmacro()
