/** @file
 * @brief Definition of class Image.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#include "edetect.hxx"
#include "Image.hxx"
#include "ImageBackend.hxx"
#include "ImageLoader.hxx"

#include "IImage.hxx"

/*************************************************************************/
/* Image                                                                 */
/*************************************************************************/
unsigned int
Image::channels(
    Image::Format fmt
    )
{
    static const unsigned int
        IMAGE_FMT_CHANNELS[] =
        {
            0, // FMT_INVALID
            1, // FMT_GRAY_UINT8
            1, // FMT_GRAY_FLOAT32
            3, // FMT_RGB_UINT8
            3, // FMT_RGB_FLOAT32
        };

    return IMAGE_FMT_CHANNELS[fmt];
}

unsigned int
Image::channelSize(
    Image::Format fmt
    )
{
    static const unsigned int
        IMAGE_FMT_CHANNEL_SIZE[] =
        {
            0,                     // FMT_INVALID
            sizeof(unsigned char), // FMT_GRAY_UINT8
            sizeof(float),         // FMT_GRAY_FLOAT32
            sizeof(unsigned char), // FMT_RGB_UINT8
            sizeof(float),         // FMT_RGB_FLOAT32
        };

    return IMAGE_FMT_CHANNEL_SIZE[fmt];
}

unsigned int
Image::pixelSize(
    Image::Format fmt
    )
{
    return channels( fmt ) * channelSize( fmt );
}

Image::Image(
    ImageBackend& backend,
    unsigned int rows,
    unsigned int cols,
    Image::Format fmt
    )
: mImage( backend.createImage() )
{
    reset( rows, cols, fmt );
}

Image::Image(
    ImageBackend& backend,
    const char* file
    )
: mImage( backend.createImage() )
{
    load( file );
}

Image::Image(
    ImageBackend& backend,
    const void* data,
    unsigned int rows,
    unsigned int cols,
    unsigned int stride,
    Image::Format fmt
    )
: mImage( backend.createImage() )
{
    load( data, rows, cols, stride, fmt );
}

Image::Image(
    const Image& oth
    )
: mImage( oth.mImage->clone() )
{
}

Image::~Image()
{
    delete mImage;
}

unsigned char*
Image::data()
{
    return mImage->data();
}

const unsigned char*
Image::data() const
{
    return mImage->data();
}

unsigned int
Image::rows() const
{
    return mImage->rows();
}

unsigned int
Image::columns() const
{
    return mImage->columns();
}

unsigned int
Image::stride() const
{
    return mImage->stride();
}

Image::Format
Image::format() const
{
    return mImage->format();
}

unsigned int
Image::channels() const
{
    return channels( mImage->format() );
}

unsigned int
Image::channelSize() const
{
    return channelSize( mImage->format() );
}

unsigned int
Image::pixelSize() const
{
    return pixelSize( mImage->format() );
}

void
Image::load(
    const char* file
    )
{
    ImageLoader::load( *mImage, file );
}

void
Image::load(
    const void* data,
    unsigned int rows,
    unsigned int cols,
    unsigned int stride,
    Image::Format fmt
    )
{
    mImage->load( data, rows, cols, stride, fmt );
}

void
Image::save(
    const char* file
    ) const
{
    ImageLoader::save( file, *mImage );
}

void
Image::save(
    void* data,
    unsigned int stride
    ) const
{
    mImage->save( data, stride );
}

void
Image::reset(
    unsigned int rows,
    unsigned int cols,
    Image::Format fmt
    )
{
    mImage->reset( rows, cols, fmt );
}

void
Image::swap(
    Image& oth
    )
{
    mImage->swap( *oth.mImage );
}
