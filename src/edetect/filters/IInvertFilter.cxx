/** @file
 * @brief Definition of class IInvertFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "filters/IInvertFilter.hxx"

/*************************************************************************/
/* IInvertFilter                                                         */
/*************************************************************************/
void
IInvertFilter::filter(
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
            "IInvertFilter: Unsupported image format" );
    }

    invert( image );
}
