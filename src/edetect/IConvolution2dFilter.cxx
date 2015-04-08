/** @file
 * @brief Definition of class IConvolution2dFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "IConvolution2dFilter.hxx"
#include "IImage.hxx"

/*************************************************************************/
/* IConvolution2dFilter                                                  */
/*************************************************************************/
IConvolution2dFilter::IConvolution2dFilter()
: mKernel( NULL ),
  mRadius( 0 )
{
}

void
IConvolution2dFilter::filter(
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
            "IConvolution2dFilter: Unsupported image format" );
    }

    IImage* output = image.cloneImpl();
    output->reset( image.rows(), image.columns(),
                   Image::FMT_GRAY_FLOAT32 );

    convolve( *output, image );
    image.swap( *output );

    delete output;
}

void
IConvolution2dFilter::setParam(
    const char* name,
    const void* value
    )
{
    if( !strcmp( name, "kernel" ) )
        setKernel( (const float*)value, mRadius );
    else if( !strcmp( name, "radius" ) )
        setKernel( mKernel, *(const unsigned int*)value );
    else
        IImageFilter::setParam( name, value );
}

void
IConvolution2dFilter::setKernel(
    const float* kernel,
    unsigned int radius
    )
{
    mKernel = kernel;
    mRadius = radius;
}
