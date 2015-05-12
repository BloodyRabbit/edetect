#
# CMake build configuration file
# Author: Jan Bobek
#

cmake_minimum_required( VERSION 2.8.6 FATAL_ERROR )
cmake_policy( VERSION 2.8.6 )

project( "edetect" )

set( BUILD_SHARED_LIBS ON
  CACHE BOOL "Build shared libraries rather than static." )

if( CMAKE_COMPILER_IS_GNUCXX )
  set( CMAKE_CXX_FLAGS
    "-ansi -Wall -Wextra ${CMAKE_CXX_FLAGS}" )
endif( CMAKE_COMPILER_IS_GNUCXX )

include_directories( "include" )
add_subdirectory( "src/edetect" )
add_subdirectory( "src/edetect-proc" )

include( "CPack" )
