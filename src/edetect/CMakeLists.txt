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
  "IConvolutionFilter.hxx"
  "IConvolutionFilter.cxx"
  "IConvolution2dSeparableFilter.hxx"
  "IConvolution2dSeparableFilter.cxx"
  "IDesaturateFilter.hxx"
  "IDesaturateFilter.cxx"
  "IIntFloatFilter.hxx"
  "IIntFloatFilter.cxx"
  "IKirschOperatorFilter.hxx"
  "IKirschOperatorFilter.txx"
  "IMarrHildrethOperatorFilter.hxx"
  "IMarrHildrethOperatorFilter.txx"
  "ISobelOperatorFilter.hxx"
  "ISobelOperatorFilter.txx"
  "IZeroCrossFilter.hxx"
  "IZeroCrossFilter.cxx"
  # Generic filter classes
  "GaussianBlurFilter.hxx"
  "GaussianBlurFilter.txx"
  "LaplacianOfGaussianFilter.hxx"
  "LaplacianOfGaussianFilter.txx"
  "SeparableConvolutionFilter.hxx"
  "SeparableConvolutionFilter.txx"

  # CPU backend implementation
  "cpu/CpuBackend.hxx"
  "cpu/CpuBackend.cxx"
  "cpu/CpuImage.hxx"
  "cpu/CpuImage.cxx"
  # CPU-backed filters
  "cpu/CpuConvolutionFilter.hxx"
  "cpu/CpuConvolutionFilter.cxx"
  "cpu/CpuConvolution2dSeparableFilter.hxx"
  "cpu/CpuConvolution2dSeparableFilter.cxx"
  "cpu/CpuDesaturateFilter.hxx"
  "cpu/CpuDesaturateFilter.cxx"
  "cpu/CpuIntFloatFilter.hxx"
  "cpu/CpuIntFloatFilter.cxx"
  "cpu/CpuKirschOperatorFilter.hxx"
  "cpu/CpuKirschOperatorFilter.cxx"
  "cpu/CpuMarrHildrethOperatorFilter.hxx"
  "cpu/CpuMarrHildrethOperatorFilter.cxx"
  "cpu/CpuSobelOperatorFilter.hxx"
  "cpu/CpuSobelOperatorFilter.cxx"
  "cpu/CpuZeroCrossFilter.hxx"
  "cpu/CpuZeroCrossFilter.cxx"

  # CUDA backend implementation
  "cuda/CudaBackend.hxx"
  "cuda/CudaBackend.cxx"
  "cuda/CudaError.hxx"
  "cuda/CudaError.cxx"
  "cuda/CudaImage.hxx"
  "cuda/CudaImage.cxx"
  # CUDA-backed filters
  "cuda/CudaConvolutionFilter.hxx"
  "cuda/CudaConvolutionFilter.cu"
  "cuda/CudaConvolution2dSeparableFilter.hxx"
  "cuda/CudaConvolution2dSeparableFilter.cu"
  "cuda/CudaDesaturateFilter.hxx"
  "cuda/CudaDesaturateFilter.cu"
  "cuda/CudaIntFloatFilter.hxx"
  "cuda/CudaIntFloatFilter.cu"
  "cuda/CudaKirschOperatorFilter.hxx"
  "cuda/CudaKirschOperatorFilter.cu"
  "cuda/CudaMarrHildrethOperatorFilter.hxx"
  "cuda/CudaMarrHildrethOperatorFilter.cu"
  "cuda/CudaSobelOperatorFilter.hxx"
  "cuda/CudaSobelOperatorFilter.cu"
  "cuda/CudaZeroCrossFilter.hxx"
  "cuda/CudaZeroCrossFilter.cu"
  )
target_link_libraries(
  "edetect"
  "freeimageplus"
  )
