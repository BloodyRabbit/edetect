/** @file
 * @brief Definition of class IKirschOperatorFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "filters/IKirschOperatorFilter.hxx"

/*************************************************************************/
/* IKirschOperatorFilter                                                 */
/*************************************************************************/
void
IKirschOperatorFilter::filter(
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
            "IKirschOperatorFilter: Unsupported image format" );
    }

    IImage* output = image.cloneImpl();
    output->reset( image.rows(), image.columns(),
                   Image::FMT_GRAY_FLOAT32 );

    applyKirschOperator( *output, image );
    image.swap( *output );
    delete output;
}
