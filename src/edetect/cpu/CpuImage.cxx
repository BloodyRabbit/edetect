/** @file
 * @brief Definition of class CpuImage.
 *
 * @author Jan Bobek
 * @since 8th April 2015
 */

#include "edetect.hxx"
#include "IImageFilter.hxx"
#include "cpu/CpuImage.hxx"

/*************************************************************************/
/* CpuImage                                                              */
/*************************************************************************/
CpuImage::~CpuImage()
{
    reset();
}

IImage*
CpuImage::clone() const
{
    IImage* res = cloneImpl();
    res->load( mData, mRows, mColumns, mStride, mFmt );
    return res;
}

IImage*
CpuImage::cloneImpl() const
{
    return new CpuImage;
}

void
CpuImage::load(
    const void* data,
    unsigned int rows,
    unsigned int cols,
    unsigned int stride,
    Image::Format fmt
    )
{
    reset( rows, cols, fmt );

    const unsigned char* data_ =
        (const unsigned char*)data;
    const unsigned int rowSize =
        cols * Image::pixelSize( fmt );

    for( unsigned int i = 0; i < rows; ++i )
        memcpy( &mData[i * mStride], &data_[i * stride], rowSize );
}

void
CpuImage::save(
    void* data,
    unsigned int stride
    ) const
{
    if( !stride )
        stride = mStride;

    unsigned char* data_ =
        (unsigned char*)data;
    const unsigned int rowSize =
        mColumns * Image::pixelSize( mFmt );

    for( unsigned int i = 0; i < mRows; ++i )
        memcpy( &data_[i * stride], &mData[i * mStride], rowSize );
}

void
CpuImage::reset(
    unsigned int rows,
    unsigned int cols,
    Image::Format fmt
    )
{
    if( mData )
        delete[] mData;

    const unsigned int rowSize =
        cols * Image::pixelSize( fmt );
    if( 0 < rows * rowSize )
    {
        mData = new unsigned char[rows * rowSize];
        mRows = rows;
        mColumns = cols;
        mStride = rowSize;
        mFmt = fmt;
    }
    else
    {
        mData = NULL;
        mRows = mColumns = mStride = 0;
        mFmt = Image::FMT_INVALID;
    }
}
