/** @file
 * @brief Definition of CudaIntFloatFilter class.
 *
 * @author Jan Bobek
 */

#include "common.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaIntFloatFilter.hxx"

/**
 * @brief CUDA kernel converting integer-pixels to float-pixels.
 *
 * @param[out] dst
 *   The destination image data.
 * @param[in] dstStride
 *   Size of the row stride in destination data.
 * @param[in] src
 *   The source image data.
 * @param[in] srcStride
 *   Size of the row stride in source data.
 * @param[in] rows
 *   Number of rows in the image.
 * @param[in] cols
 *   Number of columns in the image.
 */
__global__ void
convertInt2Float(
    unsigned char* dst,
    size_t dstStride,
    const unsigned char* src,
    size_t srcStride,
    size_t rows,
    size_t cols
    )
{
    const size_t col =
        blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if( col < cols && row < rows )
    {
        float* const dstp =
            (float*)(dst + row * dstStride) + col;
        const unsigned char* const srcp =
            src + row * srcStride + col;

        *dstp = *srcp / 255.0f;
    }
}

/**
 * @brief CUDA kernel converting float-pixels to integer-pixels.
 *
 * @param[out] dst
 *   The destination image data.
 * @param[in] dstStride
 *   Size of the row stride in destination data.
 * @param[in] src
 *   The source image data.
 * @param[in] srcStride
 *   Size of the row stride in source data.
 * @param[in] rows
 *   Number of rows in the image.
 * @param[in] cols
 *   Number of columns in the image.
 */
__global__ void
convertFloat2Int(
    unsigned char* dst,
    size_t dstStride,
    const unsigned char* src,
    size_t srcStride,
    size_t rows,
    size_t cols
    )
{
    const size_t col =
        blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if( col < cols && row < rows )
    {
        unsigned char* const dstp =
            dst + row * dstStride + col;
        const float* const srcp =
            (const float*)(src + row * srcStride) + col;

        *dstp = (unsigned char)(__saturatef(*srcp) * 255.0f);
    }
}

/*************************************************************************/
/* CudaIntFloatFilter                                                    */
/*************************************************************************/
const CudaImage::Format
CudaIntFloatFilter::FMT_TARGET[] =
{
    CudaImage::FMT_INVALID,      // FMT_INVALID
    CudaImage::FMT_GRAY_FLOAT32, // FMT_GRAY_UINT8
    CudaImage::FMT_GRAY_UINT8,   // FMT_GRAY_FLOAT32
    CudaImage::FMT_RGB_FLOAT32,  // FMT_RGB_UINT8
    CudaImage::FMT_RGB_UINT8,    // FMT_RGB_FLOAT32
};

void
CudaIntFloatFilter::process(
    CudaImage& image
    )
{
    const CudaImage::Format fmtTarget =
        FMT_TARGET[image.format()];

    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (image.columns() * image.channels() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (image.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    CudaImage newImage(
        image.rows(), image.columns(), fmtTarget );

    switch( fmtTarget )
    {
    case CudaImage::FMT_RGB_FLOAT32:
    case CudaImage::FMT_GRAY_FLOAT32:
        convertInt2Float<<< numBlocks, threadsPerBlock >>>(
            (unsigned char*)newImage.data(), newImage.rowStride(),
            (unsigned char*)image.data(), image.rowStride(),
            image.rows(), image.columns() * image.channels() );
        break;

    case CudaImage::FMT_RGB_UINT8:
    case CudaImage::FMT_GRAY_UINT8:
        convertFloat2Int<<< numBlocks, threadsPerBlock >>>(
            (unsigned char*)newImage.data(), newImage.rowStride(),
            (unsigned char*)image.data(), image.rowStride(),
            image.rows(), image.columns() * image.channels() );
        break;

    default:
    case CudaImage::FMT_INVALID:
        throw std::runtime_error(
            "CudaIntFloatFilter: invalid format" );
    }

    cudaCheckLastError( "Int-Float conversion kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "Int-Float conversion kernel run failed" );

    image.swap( newImage );
}
