/** @file
 * @brief Definition of class CpuBackend.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "cpu/CpuBackend.hxx"
#include "cpu/CpuImage.hxx"

#include "GaussianBlurFilter.hxx"
#include "LaplacianOfGaussianFilter.hxx"
#include "cpu/CpuConvolution2dFilter.hxx"
#include "cpu/CpuConvolution2dSeparableFilter.hxx"
#include "cpu/CpuDesaturateFilter.hxx"
#include "cpu/CpuIntFloatFilter.hxx"
#include "cpu/CpuKirschOperatorFilter.hxx"
#include "cpu/CpuSobelOperatorFilter.hxx"
#include "cpu/CpuZeroCrossFilter.hxx"

/*************************************************************************/
/* CpuBackend                                                            */
/*************************************************************************/
IImage*
CpuBackend::createImage()
{
    return new CpuImage;
}

IImageFilter*
CpuBackend::createFilter(
    const char* name
    )
{
    if( !strcmp( name, "conv-2d" ) )
        return new CpuConvolution2dFilter;
    if( !strcmp( name, "conv-2d-sep" ) )
        return new CpuConvolution2dSeparableFilter;
    if( !strcmp( name, "desaturate" ) )
        return new CpuDesaturateFilter;
    if( !strcmp( name, "gaussian-blur" ) )
        return new GaussianBlurFilter< CpuConvolution2dSeparableFilter >;
    if( !strcmp( name, "int-float" ) )
        return new CpuIntFloatFilter;
    if( !strcmp( name, "kirsch-operator" ) )
        return new CpuKirschOperatorFilter;
    if( !strcmp( name, "log" ) )
        return new LaplacianOfGaussianFilter< CpuConvolution2dFilter >;
    if( !strcmp( name, "sobel-operator" ) )
        return new CpuSobelOperatorFilter;
    if( !strcmp( name, "zero-cross" ) )
        return new CpuZeroCrossFilter;

    throw std::invalid_argument(
        "CpuBackend: Filter not implemented" );
}
