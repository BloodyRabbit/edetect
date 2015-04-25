/** @file
 * @brief Definition of class INonmaxSuppressionFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "filters/INonmaxSuppressionFilter.hxx"

/*************************************************************************/
/* INonmaxSuppressionFilter                                              */
/*************************************************************************/
INonmaxSuppressionFilter::INonmaxSuppressionFilter(
    IImageFilter* first,
    IImageFilter* second
    )
: IDualInputFilter( first, second )
{
}

void
INonmaxSuppressionFilter::filter(
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
            "INonmaxSuppressionFilter: Unsupported image format" );
    }

    IDualInputFilter::filter( image );
}

void
INonmaxSuppressionFilter::filter2(
    IImage& first,
    const IImage& second
    )
{
    IImage* dest = first.cloneImpl();
    dest->reset( first.rows(), first.columns(),
                 Image::FMT_GRAY_FLOAT32 );

    nonmaxSuppress( *dest, first, second );
    first.swap( *dest );
    delete dest;
}
