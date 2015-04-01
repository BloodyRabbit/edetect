#
# CMake build configuration file
# Author: Jan Bobek
#

cmake_minimum_required( VERSION 2.8.6 FATAL_ERROR )
cmake_policy( VERSION 2.8.6 )

project( "edetect" )

set( BUILD_SHARED_LIBS ON
  CACHE BOOL "Build shared libraries rather than static." )
set( CUDA_NVCC_FLAGS
  # "-gencode=arch=compute_20,code=sm_20"
  # "-gencode=arch=compute_30,code=sm_30"
  # "-gencode=arch=compute_35,code=sm_35"
  "-gencode=arch=compute_50,code=sm_50"
  ${CUDA_NVCC_FLAGS}
  CACHE STRING "Additional flags to pass to NVCC compiler." )

if( CMAKE_COMPILER_IS_GNUCXX )
  set( CMAKE_CXX_FLAGS
    "-ansi -Wall -Wextra ${CMAKE_CXX_FLAGS}" )
endif( CMAKE_COMPILER_IS_GNUCXX )

find_package( "CUDA" REQUIRED )

set(
  COMMON_SOURCES
  "common.hxx"
  "CudaConvolution2dFilter.hxx"
  "CudaConvolution2dFilter.cu"
  "CudaConvolution2dSeparableFilter.hxx"
  "CudaConvolution2dSeparableFilter.cu"
  "CudaDesaturateFilter.hxx"
  "CudaDesaturateFilter.cu"
  "CudaError.hxx"
  "CudaError.cxx"
  "CudaFilter.hxx"
  "CudaFilterPipeline.hxx"
  "CudaFilterPipeline.cxx"
  "CudaGaussianBlurFilter.hxx"
  "CudaGaussianBlurFilter.cxx"
  "CudaIntFloatFilter.hxx"
  "CudaIntFloatFilter.cu"
  "CudaImage.hxx"
  "CudaImage.cxx"
  )
set(
  COMMON_LIBRARIES
  "freeimageplus"
  )

cuda_add_executable(
  "gauss"
  ${COMMON_SOURCES}
  "gauss.cxx"
  )
target_link_libraries(
  "gauss"
  ${COMMON_LIBRARIES}
  )

cuda_add_executable(
  "kirsch"
  ${COMMON_SOURCES}
  "CudaKirschOperatorFilter.hxx"
  "CudaKirschOperatorFilter.cu"
  "kirsch.cxx"
  )
target_link_libraries(
  "kirsch"
  ${COMMON_LIBRARIES}
  )

cuda_add_executable(
  "sobel"
  ${COMMON_SOURCES}
  "CudaSobelOperatorFilter.hxx"
  "CudaSobelOperatorFilter.cu"
  "sobel.cxx"
  )
target_link_libraries(
  "sobel"
  ${COMMON_LIBRARIES}
  )
