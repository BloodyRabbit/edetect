/** @file
 * @brief Definition of class IConvolutionFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "filters/IConvolutionFilter.hxx"

/*************************************************************************/
/* IConvolutionFilter                                                    */
/*************************************************************************/
IConvolutionFilter::IConvolutionFilter()
: mKernel( NULL ),
  mLength( 0 )
{
}

IConvolutionFilter::~IConvolutionFilter()
{
    free( (void*)mKernel );
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
IConvolutionFilter::setParamVa(
    const char* name,
    va_list ap
    )
{
    const float* kernel;
    unsigned int length;

    if( !strcmp( name, "kernel" ) )
    {
        kernel = va_arg( ap, const float* );
        length = va_arg( ap, unsigned int );
        setKernel( kernel, length );
    }
    else
        IImageFilter::setParamVa( name, ap );
}

void
IConvolutionFilter::setKernel(
    const float* kernel,
    unsigned int length
    )
{
    free( (void*)mKernel );
    mKernel = kernel;
    mLength = length;
}
