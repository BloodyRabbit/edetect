/** @file
 * @brief Definition of class CudaBackend.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#include "edetect.hxx"
#include "cuda/CudaBackend.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaImage.hxx"

#include "cuda/CudaConvolutionFilter.hxx"
#include "cuda/CudaDesaturateFilter.hxx"
#include "cuda/CudaDualInputTimerFilter.hxx"
#include "cuda/CudaEuclideanNormFilter.hxx"
#include "cuda/CudaHysteresisFilter.hxx"
#include "cuda/CudaIntFloatFilter.hxx"
#include "cuda/CudaInvertFilter.hxx"
#include "cuda/CudaKirschOperatorFilter.hxx"
#include "cuda/CudaMultiplyFilter.hxx"
#include "cuda/CudaNonmaxSuppressionFilter.hxx"
#include "cuda/CudaTimerFilter.hxx"
#include "cuda/CudaZeroCrossFilter.hxx"

/*************************************************************************/
/* CudaBackend                                                           */
/*************************************************************************/
CudaBackend::CudaBackend()
{
    int device;
    cudaDeviceProp prop;

    cudaCheckError( cudaGetDevice( &device ) );
    cudaCheckError( cudaGetDeviceProperties( &prop, device ) );

    fprintf(
        stderr,
        "Active CUDA device %d: \"%s\"\n"
        "  CUDA Capability Major/Minor version number:    %d.%d\n"
        "  Number of available Multiprocessors:           %d\n"
        "  GPU Clock rate:                                %d Hz\n"
        "  Memory Clock rate:                             %d Hz\n"
        "  Memory Bus Width:                              %d-bit\n"
        "  L2 Cache Size:                                 %d bytes\n"
        "  Total amount of constant memory:               %lu bytes\n"
        "  Total amount of global memory:                 %lu bytes\n"
        "  Total amount of shared memory per block:       %lu bytes\n"
        "  Total amount of registers available per block: %d\n"
        "  Warp size:                                     %d\n"
        "  Maximum number of threads per multiprocessor:  %d\n"
        "  Maximum number of threads per block:           %d\n"
        "  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n"
        "  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n"
        "  Maximum memory pitch:                          %lu bytes\n",
        device, prop.name,
        prop.major, prop.minor,
        prop.multiProcessorCount,
        prop.clockRate,
        prop.memoryClockRate,
        prop.memoryBusWidth,
        prop.l2CacheSize,
        prop.totalConstMem,
        prop.totalGlobalMem,
        prop.sharedMemPerBlock,
        prop.regsPerBlock,
        prop.warpSize,
        prop.maxThreadsPerMultiProcessor,
        prop.maxThreadsPerBlock,
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2],
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2],
        prop.memPitch
        );
}

CudaBackend::~CudaBackend()
{
    cudaCheckError( cudaDeviceReset() );
}

IImage*
CudaBackend::createImage()
{
    return new CudaImage;
}

IImageFilter*
CudaBackend::createFilter(
    const char* name
    )
{
    if( !strcmp( name, "convolution" ) )
        return new CudaConvolutionFilter;
    if( !strcmp( name, "column-convolution" ) )
        return new CudaColumnConvolutionFilter;
    if( !strcmp( name, "desaturate" ) )
        return new CudaDesaturateFilter;
    if( !strcmp( name, "dual-input-timer" ) )
        return new CudaDualInputTimerFilter;
    if( !strcmp( name, "euclidean-norm" ) )
        return new CudaEuclideanNormFilter;
    if( !strcmp( name, "hysteresis" ) )
        return new CudaHysteresisFilter;
    if( !strcmp( name, "int-float" ) )
        return new CudaIntFloatFilter;
    if( !strcmp( name, "invert" ) )
        return new CudaInvertFilter;
    if( !strcmp( name, "kirsch-operator" ) )
        return new CudaKirschOperatorFilter;
    if( !strcmp( name, "multiply" ) )
        return new CudaMultiplyFilter;
    if( !strcmp( name, "nonmax-suppression" ) )
        return new CudaNonmaxSuppressionFilter;
    if( !strcmp( name, "row-convolution" ) )
        return new CudaRowConvolutionFilter;
    if( !strcmp( name, "timer" ) )
        return new CudaTimerFilter;
    if( !strcmp( name, "zero-cross" ) )
        return new CudaZeroCrossFilter;

    return IImageBackend::createFilter( name );
}
