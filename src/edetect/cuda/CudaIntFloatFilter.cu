/** @file
 * @brief Definition of CudaIntFloatFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaIntFloatFilter.hxx"

/**
 * @brief CUDA kernel converting integer-pixels to float-pixels.
 *
 * @param[out] ddata
 *   The destination image data.
 * @param[in] dstride
 *   Size of the row stride in destination data.
 * @param[in] sdata
 *   The source image data.
 * @param[in] sstride
 *   Size of the row stride in source data.
 * @param[in] rows
 *   Number of rows in the image.
 * @param[in] cols
 *   Number of columns in the image.
 */
__global__ void
convertInt2FloatKernel(
    unsigned char* ddata,
    unsigned int dstride,
    const unsigned char* sdata,
    unsigned int sstride,
    unsigned int rows,
    unsigned int cols
    )
{
    const unsigned int col =
        blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if( col < cols && row < rows )
    {
        float* const dstp =
            (float*)(ddata + row * dstride) + col;
        const unsigned char* const srcp =
            sdata + row * sstride + col;

        *dstp = *srcp / 255.0f;
    }
}

/**
 * @brief CUDA kernel converting float-pixels to integer-pixels.
 *
 * @param[out] ddata
 *   The destination image data.
 * @param[in] dstride
 *   Size of the row stride in destination data.
 * @param[in] sdata
 *   The source image data.
 * @param[in] sstride
 *   Size of the row stride in source data.
 * @param[in] rows
 *   Number of rows in the image.
 * @param[in] cols
 *   Number of columns in the image.
 */
__global__ void
convertFloat2IntKernel(
    unsigned char* ddata,
    unsigned int dstride,
    const unsigned char* sdata,
    unsigned int sstride,
    unsigned int rows,
    unsigned int cols
    )
{
    const unsigned int col =
        blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if( col < cols && row < rows )
    {
        unsigned char* const dstp =
            ddata + row * dstride + col;
        const float* const srcp =
            (const float*)(sdata + row * sstride) + col;

        *dstp = (unsigned char)(__saturatef(*srcp) * 255.0f);
    }
}

/*************************************************************************/
/* CudaIntFloatFilter                                                    */
/*************************************************************************/
void
CudaIntFloatFilter::convertInt2Float(
    IImage& dest,
    const IImage& src
    )
{
    const unsigned int columns =
        src.columns() * Image::channels( src.format() );

    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    convertInt2FloatKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), columns
        );

    cudaCheckLastError( "CudaIntFloatFilter: Int2Float kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaIntFloatFilter: Int2Float kernel run failed" );
}

void
CudaIntFloatFilter::convertFloat2Int(
    IImage& dest,
    const IImage& src
    )
{
    const unsigned int columns =
        src.columns() * Image::channels( src.format() );

    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    convertFloat2IntKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), columns
        );

    cudaCheckLastError( "CudaIntFloatFilter: Float2Int kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaIntFloatFilter: Float2Int kernel run failed" );
}
