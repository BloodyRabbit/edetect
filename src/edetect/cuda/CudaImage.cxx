/** @file
 * @brief Definition of CudaImage class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "IImageFilter.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaImage.hxx"

/*************************************************************************/
/* CudaImage                                                             */
/*************************************************************************/
CudaImage::CudaImage()
: mData( NULL ),
  mRows( 0 ),
  mColumns( 0 ),
  mStride( 0 ),
  mFmt( Image::FMT_INVALID )
{
}

CudaImage::~CudaImage()
{
    reset();
}

unsigned char*
CudaImage::data()
{
    return mData;
}

const unsigned char*
CudaImage::data() const
{
    return mData;
}

unsigned int
CudaImage::rows() const
{
    return mRows;
}

unsigned int
CudaImage::columns() const
{
    return mColumns;
}

unsigned int
CudaImage::stride() const
{
    return mStride;
}

Image::Format
CudaImage::format() const
{
    return mFmt;
}

void
CudaImage::load(
    const void* data,
    unsigned int rows,
    unsigned int cols,
    unsigned int stride,
    Image::Format fmt
    )
{
    reset( rows, cols, fmt );

    const unsigned int rowSize =
        cols * Image::pixelSize( fmt );
    cudaCheckError(
        cudaMemcpy2D(
            mData, mStride, data, stride,
            rowSize, rows, cudaMemcpyHostToDevice ) );
}

void
CudaImage::save(
    void* data,
    unsigned int stride
    ) const
{
    if( !stride )
        stride = mStride;

    const unsigned int rowSize =
        mColumns * Image::pixelSize( mFmt );
    cudaCheckError(
        cudaMemcpy2D(
            data, stride, mData, mStride,
            rowSize, mRows, cudaMemcpyDeviceToHost ) );
}

void
CudaImage::reset(
    unsigned int rows,
    unsigned int cols,
    Image::Format fmt
    )
{
    if( mData )
        cudaCheckError( cudaFree( mData ) );

    const unsigned int rowSize =
        cols * Image::pixelSize( fmt );
    if( 0 < rows * rowSize )
    {
        size_t stride;
        cudaCheckError(
            cudaMallocPitch(
                &mData, &stride, rowSize, rows ) );

        mRows = rows;
        mColumns = cols;
        mStride = stride;
        mFmt = fmt;
    }
    else
    {
        mData = NULL;
        mRows = mColumns = mStride = 0;
        mFmt = Image::FMT_INVALID;
    }
}

void
CudaImage::swap(
    IImage& oth
    )
{
    oth.swap( *this );
}

void
CudaImage::swap(
    CudaImage& oth
    )
{
    std::swap( mData, oth.mData );
    std::swap( mRows, oth.mRows );
    std::swap( mColumns, oth.mColumns );
    std::swap( mStride, oth.mStride );
    std::swap( mFmt, oth.mFmt );
}

CudaImage&
CudaImage::operator=(
    const CudaImage& oth
    )
{
    reset( oth.rows(), oth.columns(), oth.format() );

    const unsigned int rowSize =
        oth.columns() * Image::pixelSize( oth.format() );
    cudaCheckError(
        cudaMemcpy2D(
            mData, mStride, oth.data(), oth.stride(),
            rowSize, oth.rows(), cudaMemcpyDeviceToDevice ) );

    return *this;
}

void
CudaImage::apply(
    IImageFilter& filter
    )
{
    filter.filter( *this );
}

void
CudaImage::duplicate(
    IImage& dest
    ) const
{
    dest = *this;
}
