/** @file
 * @brief Definition of class CpuBackend.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "cpu/CpuBackend.hxx"
#include "cpu/CpuImage.hxx"

#include "cpu/CpuConvolutionFilter.hxx"
#include "cpu/CpuDesaturateFilter.hxx"
#include "cpu/CpuEuclideanNormFilter.hxx"
#include "cpu/CpuIntFloatFilter.hxx"
#include "cpu/CpuInvertFilter.hxx"
#include "cpu/CpuKirschOperatorFilter.hxx"
#include "cpu/CpuMultiplyFilter.hxx"
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
    if( !strcmp( name, "convolution" ) )
        return new CpuConvolutionFilter;
    if( !strcmp( name, "column-convolution" ) )
        return new CpuColumnConvolutionFilter;
    if( !strcmp( name, "desaturate" ) )
        return new CpuDesaturateFilter;
    if( !strcmp( name, "euclidean-norm" ) )
        return new CpuEuclideanNormFilter;
    if( !strcmp( name, "int-float" ) )
        return new CpuIntFloatFilter;
    if( !strcmp( name, "invert" ) )
        return new CpuInvertFilter;
    if( !strcmp( name, "kirsch-operator" ) )
        return new CpuKirschOperatorFilter;
    if( !strcmp( name, "multiply" ) )
        return new CpuMultiplyFilter;
    if( !strcmp( name, "row-convolution" ) )
        return new CpuRowConvolutionFilter;
    if( !strcmp( name, "zero-cross" ) )
        return new CpuZeroCrossFilter;

    return IImageBackend::createFilter( name );
}
