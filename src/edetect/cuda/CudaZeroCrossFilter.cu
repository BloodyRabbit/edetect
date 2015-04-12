/** @file
 * @brief Definition of class CudaZeroCrossFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaZeroCrossFilter.hxx"

/**
 * @brief CUDA kernel for detection
 *   of zero-crossings.
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
detectZeroCrossKernel(
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

    if( !(row < rows && col < cols) )
        return;

    float* const dstp =
        (float*)(ddata + row * dstride) + col;
    const unsigned char* const srcp =
        sdata + row * sstride + col * sizeof(float);

    const float* const tp =
        (const float*)(srcp - sstride);
    const float* const mp =
        (const float*)srcp;
    const float* const bp =
        (const float*)(srcp + sstride);

    *dstp = (1 <= row &&
             1 <= col &&
             row < (rows - 1) &&
             col < (cols - 1) &&
             (0 > mp[-1] * mp[ 1] ||
              0 > tp[ 0] * bp[ 0] ||
              0 > tp[-1] * bp[ 1] ||
              0 > tp[ 1] * bp[-1])
             ? 1.0f : 0.0f);
}

/*************************************************************************/
/* CudaZeroCrossFilter                                                   */
/*************************************************************************/
void
CudaZeroCrossFilter::detectZeroCross(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    detectZeroCrossKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaZeroCrossFilter: zero-cross kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaZeroCrossFilter: zero-cross kernel run failed" );
}
