#
# CMake build configuration file
# Author: Jan Bobek
#

set(
  COMMON_SOURCES
  "edetect-proc.hxx"
  "ImageFilterPipeline.hxx"
  "ImageFilterPipeline.cxx"
  )
set(
  COMMON_LIBRARIES
  )

add_executable(
  "gauss"
  ${COMMON_SOURCES}
  "gauss.cxx"
  )
target_link_libraries(
  "gauss"
  ${COMMON_LIBRARIES}
  )

