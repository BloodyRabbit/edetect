/** @file
 * @brief Definition of class IConvolution2dSeparableFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "IConvolution2dSeparableFilter.hxx"
#include "IImage.hxx"

/*************************************************************************/
/* IConvolution2dSeparableFilter                                         */
/*************************************************************************/
IConvolution2dSeparableFilter::IConvolution2dSeparableFilter()
: mRowKernel( NULL ),
  mRowKernelRadius( 0 ),
  mColumnKernel( NULL ),
  mColumnKernelRadius( 0 )
{
}

void
IConvolution2dSeparableFilter::filter(
    IImage& image
    )
{
    switch( image.format() )
    {
    case Image::FMT_GRAY_FLOAT32:
        break;

    default:
    case Image::FMT_GRAY_UINT8:
    case Image::FMT_RGB_UINT8:
    case Image::FMT_RGB_FLOAT32:
        throw std::runtime_error(
            "IConvolution2dSeparableFilter: Unsupported image format" );
    }

    IImage* scratch = image.cloneImpl();
    scratch->reset( image.rows(), image.columns(),
                    Image::FMT_GRAY_FLOAT32 );

    convolveRows( *scratch, image );
    convolveColumns( image, *scratch );
    delete scratch;
}

void
IConvolution2dSeparableFilter::setParam(
    const char* name,
    const void* value
    )
{
    if( !strcmp( name, "row-kernel" ) )
        setRowKernel( (const float*)value, mRowKernelRadius );
    else if( !strcmp( name, "row-kernel-radius" ) )
        setRowKernel( mRowKernel, *(const unsigned int*)value );
    else if( !strcmp( name, "column-kernel" ) )
        setColumnKernel( (const float*)value, mColumnKernelRadius );
    else if( !strcmp( name, "column-kernel-radius" ) )
        setColumnKernel( mColumnKernel, *(const unsigned int*)value );
    else
        IImageFilter::setParam( name, value );
}

void
IConvolution2dSeparableFilter::setRowKernel(
    const float* kernel,
    unsigned int radius
    )
{
    mRowKernel = kernel;
    mRowKernelRadius = radius;
}

void
IConvolution2dSeparableFilter::setColumnKernel(
    const float* kernel,
    unsigned int radius
    )
{
    mColumnKernel = kernel;
    mColumnKernelRadius = radius;
}
