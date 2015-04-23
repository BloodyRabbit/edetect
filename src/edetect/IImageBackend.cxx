/** @file
 * @brief Definition of class IImageBackend.
 *
 * @author Jan Bobek
 * @since 23th April 2015
 */

#include "edetect.hxx"
#include "IImageBackend.hxx"

#include "filters/GaussianKernel.hxx"
#include "filters/GeneratedKernelFilter.hxx"
#include "filters/SeparableConvolutionFilter.hxx"

/*************************************************************************/
/* IImageBackend                                                         */
/*************************************************************************/
IImageFilter*
IImageBackend::createFilter(
    const char* name
    )
{
    if( !strcmp( name, "row-gaussian-blur" ) )
        return new GeneratedKernelFilter< GaussianKernel >(
            createFilter( "row-convolution" ) );

    if( !strcmp( name, "column-gaussian-blur" ) )
        return new GeneratedKernelFilter< GaussianKernel >(
            createFilter( "column-convolution" ) );

    if( !strcmp( name, "gaussian-blur" ) )
        return new SeparableConvolutionFilter(
            createFilter( "row-gaussian-blur" ),
            createFilter( "column-gaussian-blur" ) );

    if( !strcmp( name, "row-derivative-of-gaussian" ) )
        return new GeneratedKernelFilter< DerivativeOfGaussianKernel >(
            createFilter( "row-convolution" ) );

    if( !strcmp( name, "column-derivative-of-gaussian" ) )
        return new GeneratedKernelFilter< DerivativeOfGaussianKernel >(
            createFilter( "column-convolution" ) );

    if( !strcmp( name, "laplacian-of-gaussian" ) )
        return new GeneratedKernelFilter< LaplacianOfGaussianKernel >(
            createFilter( "convolution" ) );

    if( !strcmp( name, "separable-convolution" ) )
        return new SeparableConvolutionFilter(
            createFilter( "row-convolution" ),
            createFilter( "column-convolution" ) );

    throw std::invalid_argument(
        "IImageBackend: Filter not implemented" );
}
