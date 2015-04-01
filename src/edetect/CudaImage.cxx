/** @file
 * @brief Definition of CudaImage class.
 *
 * @author Jan Bobek
 */

#include "common.hxx"
#include "CudaError.hxx"
#include "CudaImage.hxx"

/*************************************************************************/
/* CudaImage                                                             */
/*************************************************************************/
const size_t
CudaImage::FMT_CHANNELS[] =
{
    0, // FMT_INVALID
    1, // FMT_GRAY_UINT8
    1, // FMT_GRAY_FLOAT32
    3, // FMT_RGB_UINT8
    3, // FMT_RGB_FLOAT32
};

const size_t
CudaImage::FMT_CHANNEL_SIZE[] =
{
    0,                     // FMT_INVALID
    sizeof(unsigned char), // FMT_GRAY_UINT8
    sizeof(float),         // FMT_GRAY_FLOAT32
    sizeof(unsigned char), // FMT_RGB_UINT8
    sizeof(float),         // FMT_RGB_FLOAT32
};

CudaImage::CudaImage(
    size_t rows,
    size_t cols,
    CudaImage::Format fmt
    )
: mImage( NULL ),
  mRows( 0 ),
  mColumns( 0 ),
  mRowStride( 0 ),
  mFmt( FMT_INVALID )
{
    reset( rows, cols, fmt );
}

CudaImage::CudaImage(
    const char* file
    )
: mImage( NULL ),
  mRows( 0 ),
  mColumns( 0 ),
  mRowStride( 0 ),
  mFmt( FMT_INVALID )
{
    load( file );
}

CudaImage::CudaImage(
    const void* img,
    size_t rows,
    size_t cols,
    size_t rowStride,
    CudaImage::Format fmt
    )
: mImage( NULL ),
  mRows( 0 ),
  mColumns( 0 ),
  mRowStride( 0 ),
  mFmt( FMT_INVALID )
{
    load( img, rows, cols,
          rowStride, fmt );
}

CudaImage::CudaImage(
    const CudaImage& oth
    )
: mImage( NULL ),
  mRows( 0 ),
  mColumns( 0 ),
  mRowStride( 0 ),
  mFmt( FMT_INVALID )
{
    *this = oth;
}

CudaImage::~CudaImage()
{
    reset();
}

void
CudaImage::load(
    const char* file
    )
{
    fipImage img;
    if( !img.load( file ) )
        throw std::runtime_error( "Failed to load image from file" );

    switch( img.getColorType() )
    {
    case FIC_MINISBLACK:
        if( 8 == img.getBitsPerPixel() )
        {
            // Grayscale 8bpp
            load( img.accessPixels(), img.getHeight(),
                  img.getWidth(), img.getScanWidth(),
                  FMT_GRAY_UINT8 );
            return;
        }
        break;

    case FIC_RGB:
        if( 24 == img.getBitsPerPixel() )
        {
            // RGB 24bpp
            load( img.accessPixels(), img.getHeight(),
                  img.getWidth(), img.getScanWidth(),
                  FMT_RGB_UINT8 );
            return;
        }
        break;

    default:
        break;
    }

    throw std::runtime_error(
        "Failed to load image: Unsupported image format" );
}

void
CudaImage::load(
    const void* img,
    size_t rows,
    size_t cols,
    size_t rowStride,
    CudaImage::Format fmt
    )
{
    reset( rows, cols, fmt );

    const size_t rowSize = cols * pixelSize( fmt );
    cudaCheckError(
        cudaMemcpy2D(
            mImage, mRowStride, img, rowStride,
            rowSize, rows, cudaMemcpyHostToDevice ) );
}

void
CudaImage::save(
    const char* file
    )
{
    fipImage img;

    switch( mFmt )
    {
    case FMT_GRAY_UINT8:
        img.setSize( FIT_BITMAP, mColumns, mRows, 8 );
        break;

    case FMT_RGB_UINT8:
        img.setSize( FIT_BITMAP, mColumns, mRows, 24 );
        break;

    default:
        throw std::runtime_error(
            "Failed to save image: Unsupported image format" );
    }

    save( img.accessPixels(), img.getScanWidth() );
    if( !img.save( file ) )
        throw std::runtime_error( "Failed to save image to file" );
}

void
CudaImage::save(
    void* img,
    size_t rowStride
    )
{
    if( !rowStride )
        rowStride = mRowStride;

    const size_t rowSize = mColumns * pixelSize();
    cudaCheckError(
        cudaMemcpy2D(
            img, rowStride, mImage, mRowStride,
            rowSize, mRows, cudaMemcpyDeviceToHost ) );
}

void
CudaImage::swap(
    CudaImage& oth
    )
{
    std::swap( mImage, oth.mImage );
    std::swap( mRows, oth.mRows );
    std::swap( mColumns, oth.mColumns );
    std::swap( mRowStride, oth.mRowStride );
    std::swap( mFmt, oth.mFmt );
}

void
CudaImage::reset(
    size_t rows,
    size_t cols,
    CudaImage::Format fmt
    )
{
    if( mImage )
        cudaCheckError( cudaFree( mImage ) );

    const size_t rowSize = cols * pixelSize( fmt );
    if( 0 < rows * rowSize )
    {
        cudaCheckError(
            cudaMallocPitch(
                &mImage, &mRowStride, rowSize, rows ) );

        mRows = rows;
        mColumns = cols;
        mFmt = fmt;
    }
    else
    {
        mImage = NULL;
        mRows = mColumns = mRowStride = 0;
        mFmt = FMT_INVALID;
    }
}

CudaImage&
CudaImage::operator=(
    const CudaImage& oth
    )
{
    reset( oth.rows(), oth.columns(), oth.format() );

    const size_t rowSize = oth.columns() * oth.pixelSize();
    cudaCheckError(
        cudaMemcpy2D(
            mImage, mRowStride, oth.data(), oth.rowStride(),
            rowSize, oth.rows(), cudaMemcpyDeviceToDevice ) );

    return *this;
}
