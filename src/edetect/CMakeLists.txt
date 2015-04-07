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
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  )

cuda_add_library(
  "edetect"
  "edetect.hxx"
  "IImage.hxx"
  "IImage.ixx"
  "IImageBackend.hxx"
  "IImageFilter.hxx"
  "IImageFilter.ixx"
  "cuda/CudaConvolution2dFilter.hxx"
  "cuda/CudaConvolution2dFilter.cu"
  "cuda/CudaConvolution2dSeparableFilter.hxx"
  "cuda/CudaConvolution2dSeparableFilter.cu"
  "cuda/CudaDesaturateFilter.hxx"
  "cuda/CudaDesaturateFilter.cu"
  "cuda/CudaError.hxx"
  "cuda/CudaError.cxx"
  "cuda/CudaFilter.hxx"
  "cuda/CudaGaussianBlurFilter.hxx"
  "cuda/CudaGaussianBlurFilter.cxx"
  "cuda/CudaIntFloatFilter.hxx"
  "cuda/CudaIntFloatFilter.cu"
  "cuda/CudaImage.hxx"
  "cuda/CudaImage.cxx"
  )
target_link_libraries(
  "edetect"
  "freeimageplus"
  )
