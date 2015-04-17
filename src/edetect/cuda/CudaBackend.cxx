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
#include "SeparableConvolutionFilter.hxx"
#include "cuda/CudaConvolutionFilter.hxx"
#include "cuda/CudaDesaturateFilter.hxx"
#include "cuda/CudaIntFloatFilter.hxx"
#include "cuda/CudaKirschOperatorFilter.hxx"
#include "cuda/CudaMarrHildrethOperatorFilter.hxx"
#include "cuda/CudaSobelOperatorFilter.hxx"
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
    if( !strcmp( name, "gaussian-blur" ) )
        return new SeparableConvolutionFilter<
            GaussianBlurFilter< CudaRowConvolutionFilter >,
            GaussianBlurFilter< CudaColumnConvolutionFilter > >;
    if( !strcmp( name, "int-float" ) )
        return new CudaIntFloatFilter;
    if( !strcmp( name, "kirsch-operator" ) )
        return new CudaKirschOperatorFilter;
    if( !strcmp( name, "laplacian-of-gaussian" ) )
        return new LaplacianOfGaussianFilter<
            CudaConvolutionFilter >;
    if( !strcmp( name, "marr-hildreth" ) )
        return new CudaMarrHildrethOperatorFilter;
    if( !strcmp( name, "row-convolution" ) )
        return new CudaRowConvolutionFilter;
    if( !strcmp( name, "sobel-operator" ) )
        return new CudaSobelOperatorFilter;
    if( !strcmp( name, "zero-cross" ) )
        return new CudaZeroCrossFilter;

    throw std::invalid_argument(
        "CudaBackend: Filter not implemented" );
}
