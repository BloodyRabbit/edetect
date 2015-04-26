/** @file
 * @brief Definition of class IHysteresisFilter.
 *
 * @author Jan Bobek
 * @since 26th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "filters/IHysteresisFilter.hxx"

/*************************************************************************/
/* IHysteresisFilter                                                     */
/*************************************************************************/
void
IHysteresisFilter::filter(
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
            "IHysteresisFilter: Unsupported image format" );
    }

    IImage* dest = image.cloneImpl();
    dest->reset( image.rows(), image.columns(),
                 Image::FMT_GRAY_FLOAT32 );

    hysteresis( *dest, image );
    image.swap( *dest );
    delete dest;
}

void
IHysteresisFilter::setParamVa(
    const char* name,
    va_list ap
    )
{
    double threshold;

    if( !strcmp( name, "low-threshold" ) )
    {
        threshold = va_arg( ap, double );
        setThresholdLow( threshold );
    }
    else if( !strcmp( name, "high-threshold" ) )
    {
        threshold = va_arg( ap, double );
        setThresholdHigh( threshold );
    }
    else
        IImageFilter::setParamVa( name, ap );
}

void
IHysteresisFilter::setThresholdLow(
    float threshold
    )
{
    mThresholdLow = threshold;
}

void
IHysteresisFilter::setThresholdHigh(
    float threshold
    )
{
    mThresholdHigh = threshold;
}
