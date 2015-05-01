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
