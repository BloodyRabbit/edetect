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
  "cuda/CudaBackend.hxx"
  "cuda/CudaBackend.cxx"
  "cuda/CudaConvolution2dFilter.hxx"
  "cuda/CudaConvolution2dFilter.cu"
  "cuda/CudaConvolution2dSeparableFilter.hxx"
  "cuda/CudaConvolution2dSeparableFilter.cu"
  "cuda/CudaDesaturateFilter.hxx"
  "cuda/CudaDesaturateFilter.cu"
  "cuda/CudaError.hxx"
  "cuda/CudaError.cxx"
  "cuda/CudaGaussianBlurFilter.hxx"
  "cuda/CudaGaussianBlurFilter.cxx"
  "cuda/CudaIntFloatFilter.hxx"
  "cuda/CudaIntFloatFilter.cu"
  "cuda/CudaImage.hxx"
  "cuda/CudaImage.cxx"
  "cuda/CudaKirschOperatorFilter.hxx"
  "cuda/CudaKirschOperatorFilter.cu"
  "cuda/CudaSobelOperatorFilter.hxx"
  "cuda/CudaSobelOperatorFilter.cu"
  )
target_link_libraries(
  "edetect"
  "freeimageplus"
  )
