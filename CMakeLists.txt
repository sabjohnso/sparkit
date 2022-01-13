cmake_minimum_required(VERSION 3.12)

project(sparkit VERSION 0.1.0 LANGUAGES C CXX)

option(sparkit_BUILD_TESTING "Build the sparkit tests" ON)

include(GNUInstallDirs)
set(sparkit_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
set(sparkit_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set(sparkit_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
set(sparkit_INSTALL_CONFDIR ${sparkit_INSTALL_LIBDIR}/cmake/sparkit-${PROJECT_VERSION})

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)
include(sparkit_deps)

add_subdirectory(sparkit)

if(sparkit_BUILD_TESTING)
  include(CTest)
  if(BUILD_TESTING)
    add_subdirectory(sparkit_testing)
  endif()
endif()

install(EXPORT sparkit_EXPORTS
  NAMESPACE sparkit::
  FILE sparkit-exports.cmake
  DESTINATION ${sparkit_INSTALL_CONFDIR})

configure_file(sparkit-config.cmake.in sparkit-config.cmake)
install(FILES ${PROJECT_BINARY_DIR}/sparkit-config.cmake
  DESTINATION ${sparkit_INSTALL_CONFDIR})








