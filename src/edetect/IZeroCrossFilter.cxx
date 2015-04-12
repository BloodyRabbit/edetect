/** @file
 * @brief Definition of class IZeroCrossFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "IZeroCrossFilter.hxx"

/*************************************************************************/
/* IZeroCrossFilter                                                      */
/*************************************************************************/
void
IZeroCrossFilter::filter(
    IImage& image
    )
{
    switch( image.format() )
    {
    case Image::FMT_GRAY_FLOAT32:
        break;

    default:
    case Image::FMT_GRAY_UINT8:
    case Image::FMT_RGB_FLOAT32:
    case Image::FMT_RGB_UINT8:
        throw std::runtime_error(
            "IZeroCrossFilter: Unsupported image format" );
    }

    IImage* output = image.cloneImpl();
    output->reset( image.rows(), image.columns(),
                   Image::FMT_GRAY_FLOAT32 );

    detectZeroCross( *output, image );

    image.swap( *output );
    delete output;
}
