#
# CMake build configuration file
# Author: Jan Bobek
#

set( CUDA_NVCC_FLAGS
  # "-gencode=arch=compute_20,code=sm_20"
  # "-gencode=arch=compute_30,code=sm_30"
  # "-gencode=arch=compute_35,code=sm_35"
  "-gencode=arch=compute_50,code=sm_50"
  CACHE STRING "Additional flags to pass to NVCC compiler." )

find_package( "CUDA" REQUIRED )

cuda_add_library(
  "edetect"
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
  "CudaGaussianBlurFilter.hxx"
  "CudaGaussianBlurFilter.cxx"
  "CudaIntFloatFilter.hxx"
  "CudaIntFloatFilter.cu"
  "CudaImage.hxx"
  "CudaImage.cxx"
  )
target_link_libraries(
  "edetect"
  "freeimageplus"
  )
