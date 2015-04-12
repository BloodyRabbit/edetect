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

#include "GaussianBlurFilter.hxx"
#include "LaplacianOfGaussianFilter.hxx"
#include "cuda/CudaConvolution2dFilter.hxx"
#include "cuda/CudaConvolution2dSeparableFilter.hxx"
#include "cuda/CudaDesaturateFilter.hxx"
#include "cuda/CudaIntFloatFilter.hxx"
#include "cuda/CudaKirschOperatorFilter.hxx"
#include "cuda/CudaSobelOperatorFilter.hxx"

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
    if( !strcmp( name, "conv-2d" ) )
        return new CudaConvolution2dFilter;
    if( !strcmp( name, "conv-2d-sep" ) )
        return new CudaConvolution2dSeparableFilter;
    if( !strcmp( name, "desaturate" ) )
        return new CudaDesaturateFilter;
    if( !strcmp( name, "gaussian-blur" ) )
        return new GaussianBlurFilter< CudaConvolution2dSeparableFilter >;
    if( !strcmp( name, "int-float" ) )
        return new CudaIntFloatFilter;
    if( !strcmp( name, "kirsch-operator" ) )
        return new CudaKirschOperatorFilter;
    if( !strcmp( name, "log" ) )
        return new LaplacianOfGaussianFilter< CudaConvolution2dFilter >;
    if( !strcmp( name, "sobel-operator" ) )
        return new CudaSobelOperatorFilter;

    throw std::invalid_argument(
        "CudaBackend: Filter not implemented" );
}
