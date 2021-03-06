#
# CMake build configuration file
# Author: Jan Bobek
#

set( CUDA_NVCC_FLAGS
  "-gencode=arch=compute_20,code=sm_20"
  # "-gencode=arch=compute_30,code=sm_30"
  "-gencode=arch=compute_35,code=sm_35"
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
  "ImageFilterBuilder.cxx"
  "ImageLoader.hxx"
  "ImageLoader.cxx"
  "ImageFilterPipeline.hxx"
  "ImageFilterPipeline.cxx"
  "StringFilterBuilderImpl.hxx"
  "StringFilterBuilderImpl.cxx"
  "XmlFilterBuilderImpl.hxx"
  "XmlFilterBuilderImpl.cxx"
  # Internal interface classes
  "IImage.hxx"
  "IImage.ixx"
  "IImageBackend.hxx"
  "IImageBackend.cxx"
  "IImageFilter.hxx"
  "IImageFilter.ixx"
  "IImageFilterBuilder.hxx"

  # Filter interface classes
  "filters/IConvolutionFilter.hxx"
  "filters/IConvolutionFilter.cxx"
  "filters/IDesaturateFilter.hxx"
  "filters/IDesaturateFilter.cxx"
  "filters/IDualInputFilter.hxx"
  "filters/IDualInputFilter.cxx"
  "filters/IDualInputTimerFilter.hxx"
  "filters/IDualInputTimerFilter.cxx"
  "filters/IEuclideanNormFilter.hxx"
  "filters/IEuclideanNormFilter.cxx"
  "filters/IHysteresisFilter.hxx"
  "filters/IHysteresisFilter.cxx"
  "filters/IIntFloatFilter.hxx"
  "filters/IIntFloatFilter.cxx"
  "filters/IInvertFilter.hxx"
  "filters/IInvertFilter.cxx"
  "filters/IKirschOperatorFilter.hxx"
  "filters/IKirschOperatorFilter.cxx"
  "filters/INonmaxSuppressionFilter.hxx"
  "filters/INonmaxSuppressionFilter.cxx"
  "filters/IMultiplyFilter.hxx"
  "filters/IMultiplyFilter.cxx"
  "filters/ITimerFilter.hxx"
  "filters/ITimerFilter.cxx"
  "filters/IZeroCrossFilter.hxx"
  "filters/IZeroCrossFilter.cxx"
  # Generic filter classes
  "filters/GaussianKernel.hxx"
  "filters/GaussianKernel.cxx"
  "filters/GeneratedKernelFilter.hxx"
  "filters/GeneratedKernelFilter.txx"
  "filters/SeparableConvolutionFilter.hxx"
  "filters/SeparableConvolutionFilter.cxx"

  # CPU backend implementation
  "cpu/CpuBackend.hxx"
  "cpu/CpuBackend.cxx"
  "cpu/CpuImage.hxx"
  "cpu/CpuImage.cxx"
  # CPU-backed filters
  "cpu/CpuConvolutionFilter.hxx"
  "cpu/CpuConvolutionFilter.cxx"
  "cpu/CpuDesaturateFilter.hxx"
  "cpu/CpuDesaturateFilter.cxx"
  "cpu/CpuDualInputTimerFilter.hxx"
  "cpu/CpuDualInputTimerFilter.cxx"
  "cpu/CpuEuclideanNormFilter.hxx"
  "cpu/CpuEuclideanNormFilter.cxx"
  "cpu/CpuHysteresisFilter.hxx"
  "cpu/CpuHysteresisFilter.cxx"
  "cpu/CpuIntFloatFilter.hxx"
  "cpu/CpuIntFloatFilter.cxx"
  "cpu/CpuInvertFilter.hxx"
  "cpu/CpuInvertFilter.cxx"
  "cpu/CpuKirschOperatorFilter.hxx"
  "cpu/CpuKirschOperatorFilter.cxx"
  "cpu/CpuMultiplyFilter.hxx"
  "cpu/CpuMultiplyFilter.cxx"
  "cpu/CpuNonmaxSuppressionFilter.hxx"
  "cpu/CpuNonmaxSuppressionFilter.cxx"
  "cpu/CpuTimerFilter.hxx"
  "cpu/CpuTimerFilter.cxx"
  "cpu/CpuZeroCrossFilter.hxx"
  "cpu/CpuZeroCrossFilter.cxx"

  # CUDA backend implementation
  "cuda/CudaBackend.hxx"
  "cuda/CudaBackend.cxx"
  "cuda/CudaError.hxx"
  "cuda/CudaError.cxx"
  "cuda/CudaImage.hxx"
  "cuda/CudaImage.cxx"
  "cuda/CudaRingBuffer.cuh"
  "cuda/CudaRingBuffer.cut"
  # CUDA-backed filters
  "cuda/CudaConvolutionFilter.hxx"
  "cuda/CudaConvolutionFilter.cu"
  "cuda/CudaDesaturateFilter.hxx"
  "cuda/CudaDesaturateFilter.cu"
  "cuda/CudaDualInputTimerFilter.hxx"
  "cuda/CudaDualInputTimerFilter.cxx"
  "cuda/CudaEuclideanNormFilter.hxx"
  "cuda/CudaEuclideanNormFilter.cu"
  "cuda/CudaHysteresisFilter.hxx"
  "cuda/CudaHysteresisFilter.cu"
  "cuda/CudaIntFloatFilter.hxx"
  "cuda/CudaIntFloatFilter.cu"
  "cuda/CudaInvertFilter.hxx"
  "cuda/CudaInvertFilter.cu"
  "cuda/CudaKirschOperatorFilter.hxx"
  "cuda/CudaKirschOperatorFilter.cu"
  "cuda/CudaMultiplyFilter.hxx"
  "cuda/CudaMultiplyFilter.cu"
  "cuda/CudaNonmaxSuppressionFilter.hxx"
  "cuda/CudaNonmaxSuppressionFilter.cu"
  "cuda/CudaTimerFilter.hxx"
  "cuda/CudaTimerFilter.cxx"
  "cuda/CudaZeroCrossFilter.hxx"
  "cuda/CudaZeroCrossFilter.cu"
  )
target_link_libraries(
  "edetect"
  "freeimageplus"
  "tinyxml2"
  )
install(
  TARGETS "edetect"
  RUNTIME DESTINATION "bin"
  LIBRARY DESTINATION "lib"
  )
