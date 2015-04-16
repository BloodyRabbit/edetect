/** @file
 * @brief Definition of class IConvolutionFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "IConvolutionFilter.hxx"
#include "IImage.hxx"

/*************************************************************************/
/* IConvolutionFilter                                                    */
/*************************************************************************/
IConvolutionFilter::IConvolutionFilter()
: mKernel( NULL ),
  mRadius( 0 )
{
}

void
IConvolutionFilter::filter(
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
            "IConvolutionFilter: Unsupported image format" );
    }

    IImage* output = image.cloneImpl();
    output->reset( image.rows(), image.columns(),
                   Image::FMT_GRAY_FLOAT32 );

    convolve( *output, image );
    image.swap( *output );

    delete output;
}

void
IConvolutionFilter::setParam(
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
IConvolutionFilter::setKernel(
    const float* kernel,
    unsigned int radius
    )
{
    mKernel = kernel;
    mRadius = radius;
}
