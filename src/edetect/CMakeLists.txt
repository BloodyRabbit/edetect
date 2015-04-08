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

  # Public API implementation
  "Image.cxx"
  "ImageBackend.cxx"
  "ImageFilter.cxx"
  "ImageLoader.hxx"
  "ImageLoader.cxx"
  # Internal interface classes
  "IImage.hxx"
  "IImage.ixx"
  "IImageBackend.hxx"
  "IImageFilter.hxx"
  "IImageFilter.ixx"
  # Filter interface classes
  "IConvolution2dFilter.hxx"
  "IConvolution2dFilter.cxx"
  "IConvolution2dSeparableFilter.hxx"
  "IConvolution2dSeparableFilter.cxx"
  "IDesaturateFilter.hxx"
  "IDesaturateFilter.cxx"
  "IIntFloatFilter.hxx"
  "IIntFloatFilter.cxx"
  "IKirschOperatorFilter.hxx"
  "IKirschOperatorFilter.txx"
  "ISobelOperatorFilter.hxx"
  "ISobelOperatorFilter.txx"
  # Generic filter classes
  "GaussianBlurFilter.hxx"
  "GaussianBlurFilter.txx"

  # CPU backend implementation
  "cpu/CpuBackend.hxx"
  "cpu/CpuBackend.cxx"
  "cpu/CpuImage.hxx"
  "cpu/CpuImage.cxx"
  # CPU-backed filters
  "cpu/CpuConvolution2dFilter.hxx"
  "cpu/CpuConvolution2dFilter.cxx"
  "cpu/CpuConvolution2dSeparableFilter.hxx"
  "cpu/CpuConvolution2dSeparableFilter.cxx"
  "cpu/CpuDesaturateFilter.hxx"
  "cpu/CpuDesaturateFilter.cxx"
  "cpu/CpuIntFloatFilter.hxx"
  "cpu/CpuIntFloatFilter.cxx"
  "cpu/CpuKirschOperatorFilter.hxx"
  "cpu/CpuKirschOperatorFilter.cxx"
  "cpu/CpuSobelOperatorFilter.hxx"
  "cpu/CpuSobelOperatorFilter.cxx"

  # CUDA backend implementation
  "cuda/CudaBackend.hxx"
  "cuda/CudaBackend.cxx"
  "cuda/CudaError.hxx"
  "cuda/CudaError.cxx"
  "cuda/CudaImage.hxx"
  "cuda/CudaImage.cxx"
  # CUDA-backed filters
  "cuda/CudaConvolution2dFilter.hxx"
  "cuda/CudaConvolution2dFilter.cu"
  "cuda/CudaConvolution2dSeparableFilter.hxx"
  "cuda/CudaConvolution2dSeparableFilter.cu"
  "cuda/CudaDesaturateFilter.hxx"
  "cuda/CudaDesaturateFilter.cu"
  "cuda/CudaIntFloatFilter.hxx"
  "cuda/CudaIntFloatFilter.cu"
  "cuda/CudaKirschOperatorFilter.hxx"
  "cuda/CudaKirschOperatorFilter.cu"
  "cuda/CudaSobelOperatorFilter.hxx"
  "cuda/CudaSobelOperatorFilter.cu"
  )
target_link_libraries(
  "edetect"
  "freeimageplus"
  )
