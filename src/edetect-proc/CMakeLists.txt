#
# CMake build configuration file
# Author: Jan Bobek
#

set(
  COMMON_SOURCES
  )
set(
  COMMON_LIBRARIES
  "edetect"
  )

add_executable(
  "edetect-proc"
  "edetect-proc.hxx"
  "edetect-proc.cxx"
  )
target_link_libraries(
  "edetect-proc"
  "edetect"
  )
install(
  TARGETS "edetect-proc"
  RUNTIME DESTINATION "bin"
  )
