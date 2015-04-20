/** @file
 * @brief Definition of CudaKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaKirschOperatorFilter.hxx"

/**
 * @brief CUDA kernel for applying
 *   the Kirsch operator.
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
applyKirschOperatorKernel(
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
        (const float*)(0 < row ? srcp - sstride : srcp);
    const float* const mp =
        (const float*)srcp;
    const float* const bp =
        (const float*)(row + 1 < rows ? srcp + sstride : srcp);

    const int li = (0 < col ? -1 : 0);
    const int ri = (col + 1 < cols ? 1 : 0);

    float x =
        5.0f * (tp[li] + tp[ 0] + tp[ri]) -
        3.0f * (mp[li] + mp[ri] + bp[li] + bp[0] + bp[ri]);

    float a = fabs(x);
    a = fmaxf( a, fabs( x += 8.0f * (mp[ri] - tp[li]) ) );
    a = fmaxf( a, fabs( x += 8.0f * (bp[ri] - tp[ 0]) ) );
    a = fmaxf( a, fabs( x += 8.0f * (bp[ 0] - tp[ri]) ) );
    a = fmaxf( a, fabs( x += 8.0f * (bp[li] - mp[ri]) ) );
    a = fmaxf( a, fabs( x += 8.0f * (mp[li] - bp[ri]) ) );
    a = fmaxf( a, fabs( x += 8.0f * (tp[li] - bp[ 0]) ) );
    a = fmaxf( a, fabs( x += 8.0f * (tp[ 0] - bp[li]) ) );

    *dstp = a;
}

/*************************************************************************/
/* CudaKirschOperatorFilter                                              */
/*************************************************************************/
void
CudaKirschOperatorFilter::applyKirschOperator(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    applyKirschOperatorKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaKirschOperatorFilter: kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaKirschOperatorFilter: kernel run failed" );
}
