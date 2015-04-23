/** @file
 * @brief Definition of class IIntFloatFilter.
 *
 * @author Jan Bobek
 * @since 9th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "filters/IIntFloatFilter.hxx"

/*************************************************************************/
/* IIntFloatFilter                                                       */
/*************************************************************************/
const Image::Format
IIntFloatFilter::FMT_TARGET[] =
{
    Image::FMT_INVALID,      // FMT_INVALID
    Image::FMT_GRAY_FLOAT32, // FMT_GRAY_UINT8
    Image::FMT_GRAY_UINT8,   // FMT_GRAY_FLOAT32
    Image::FMT_RGB_FLOAT32,  // FMT_RGB_UINT8
    Image::FMT_RGB_UINT8,    // FMT_RGB_FLOAT32
};

void
IIntFloatFilter::filter(
    IImage& image
    )
{
    IImage* output = image.cloneImpl();
    output->reset( image.rows(), image.columns(),
                   FMT_TARGET[image.format()] );

    switch( output->format() )
    {
    case Image::FMT_RGB_FLOAT32:
    case Image::FMT_GRAY_FLOAT32:
        convertInt2Float( *output, image );
        break;

    case Image::FMT_RGB_UINT8:
    case Image::FMT_GRAY_UINT8:
        convertFloat2Int( *output, image );
        break;

    default:
    case Image::FMT_INVALID:
        throw std::runtime_error(
            "IIntFloatFilter: invalid format" );
    }

    image.swap( *output );
    delete output;
}
