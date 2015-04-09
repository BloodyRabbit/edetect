/** @file
 * @brief Definition of CudaImage class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaImage.hxx"

/*************************************************************************/
/* CudaImage                                                             */
/*************************************************************************/
CudaImage::~CudaImage()
{
    reset();
}

IImage*
CudaImage::clone() const
{
    IImage* res = cloneImpl();
    res->reset( mRows, mColumns, mFmt );

    const unsigned int rowSize =
        mColumns * Image::pixelSize( mFmt );
    cudaCheckError(
        cudaMemcpy2D(
            res->data(), res->stride(), mData, mStride,
            rowSize, mRows, cudaMemcpyDeviceToDevice ) );

    return res;
}

IImage*
CudaImage::cloneImpl() const
{
    return new CudaImage;
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
