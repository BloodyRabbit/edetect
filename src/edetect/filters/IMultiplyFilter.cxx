/** @file
 * @brief Definition of class IMultiplyFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "filters/IMultiplyFilter.hxx"

/*************************************************************************/
/* IMultiplyFilter                                                       */
/*************************************************************************/
IMultiplyFilter::IMultiplyFilter(
    IImageFilter* first,
    IImageFilter* second
    )
: IDualInputFilter( first, second )
{
}

void
IMultiplyFilter::filter(
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
            "IMultiplyFilter: Unsupported image format" );
    }

    IDualInputFilter::filter( image );
}
