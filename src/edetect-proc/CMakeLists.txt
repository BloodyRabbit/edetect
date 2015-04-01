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

add_executable( "gauss" "gauss.cxx" ${COMMON_SOURCES} )
target_link_libraries( "gauss" ${COMMON_LIBRARIES} )

add_executable( "sobel" "sobel.cxx" ${COMMON_SOURCES} )
target_link_libraries( "sobel" ${COMMON_LIBRARIES} )
