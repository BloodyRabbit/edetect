/** @file
 * @brief Definition of class ImageLoader.
 *
 * @author Jan Bobek
 * @since 8th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "ImageLoader.hxx"

/*************************************************************************/
/* ImageLoader                                                           */
/*************************************************************************/
void
ImageLoader::load(
    IImage& dest,
    const char* file
    )
{
    fipImage img;
    if( !img.load( file ) )
        throw std::runtime_error( "ImageLoader: Failed to load image from file" );

    switch( img.getColorType() )
    {
    case FIC_MINISBLACK:
        if( 8 == img.getBitsPerPixel() )
        {
            // Grayscale 8bpp
            dest.load( img.accessPixels(), img.getHeight(),
                       img.getWidth(), img.getScanWidth(),
                       Image::FMT_GRAY_UINT8 );
            return;
        }
        break;

    case FIC_RGB:
        if( 24 == img.getBitsPerPixel() )
        {
            // RGB 24bpp
            dest.load( img.accessPixels(), img.getHeight(),
                       img.getWidth(), img.getScanWidth(),
                       Image::FMT_RGB_UINT8 );
            return;
        }
        break;

    default:
        break;
    }

    throw std::runtime_error(
        "ImageLoader: Cannot load unsupported image format" );
}

void
ImageLoader::save(
    const char* file,
    const IImage& image
    )
{
    fipImage img;

    switch( image.format() )
    {
    case Image::FMT_GRAY_UINT8:
        img.setSize( FIT_BITMAP, image.columns(), image.rows(), 8 );
        break;

    case Image::FMT_RGB_UINT8:
        img.setSize( FIT_BITMAP, image.columns(), image.rows(), 24 );
        break;

    default:
        throw std::runtime_error(
            "ImageLoader: Cannot save unsupported image format" );
    }

    image.save( img.accessPixels(), img.getScanWidth() );
    if( !img.save( file ) )
        throw std::runtime_error( "ImageLoader: Failed to save image to file" );
}
