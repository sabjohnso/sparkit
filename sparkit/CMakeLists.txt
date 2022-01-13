configure_file(config.hpp.in config.hpp)

add_library(sparkit_header INTERFACE)
set_target_properties(sparkit_header PROPERTIES EXPORT_NAME header)
target_include_directories(sparkit_header
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${sparkit_INSTALL_INCLUDEDIR}>)

add_library(sparkit::header ALIAS sparkit_header)
  
install(TARGETS sparkit_header
  EXPORT sparkit_EXPORTS
  RUNTIME DESTINATION ${sparkit_INSTALL_BINDIR}
  LIBRARY DESTINATION ${sparkit_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${sparkit_INSTALL_LIBDIR})

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/config.hpp
  DESTINATION ${sparkit_INSTALL_INCLUDEDIR}/sparkit)
