/** @file
 * @brief Definition of class CudaNonmaxSuppressionFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaNonmaxSuppressionFilter.hxx"

/**
 * @brief CUDA kernel performing non-maximum suppression.
 *
 * @param[out] ddata
 *   The destination image data.
 * @param[in] dstride
 *   Size of the row stride in destination data.
 * @param[in] gxdata
 *   The X-gradient data.
 * @param[in] gxstride
 *   Size of the row stride in X-gradient data.
 * @param[in] gydata
 *   The Y-gradient data.
 * @param[in] gystride
 *   Size of the row stride in Y-gradient data.
 * @param[in] rows
 *   Number of rows in the image.
 * @param[in] cols
 *   Number of columns in the image.
 */
__global__ void
nonmaxSuppressKernel(
    unsigned char* ddata,
    unsigned int dstride,
    const unsigned char* gxdata,
    unsigned int gxstride,
    const unsigned char* gydata,
    unsigned int gystride,
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

    const unsigned char* const gxp =
        gxdata + row * gxstride + col * sizeof(float);
    const unsigned char* const gyp =
        gydata + row * gystride + col * sizeof(float);

    const float* const gxtp =
        (const float*)(gxp - gxstride);
    const float* const gxmp =
        (const float*)gxp;
    const float* const gxbp =
        (const float*)(gxp + gxstride);

    const float* const gytp =
        (const float*)(gyp - gystride);
    const float* const gymp =
        (const float*)gyp;
    const float* const gybp =
        (const float*)(gyp + gystride);

    const float gx = gxmp[0];
    const float gy = gymp[0];
    float gm = sqrtf( gx * gx + gy * gy );

    if( 0 < row && row < (rows - 1) &&
        0 < col && col < (cols - 1) &&
        0.0f < gm )
    {
        float q, gm1, gm2, gm3, gm4;

        if( 0 < gx * gy )
            (gm1 = sqrtf( gxtp[-1] * gxtp[-1] + gytp[-1] * gytp[-1] ),
             gm3 = sqrtf( gxbp[ 1] * gxbp[ 1] + gybp[ 1] * gybp[ 1] ));
        else
            (gm1 = sqrtf( gxtp[ 1] * gxtp[ 1] + gytp[ 1] * gytp[ 1] ),
             gm3 = sqrtf( gxbp[-1] * gxbp[-1] + gybp[-1] * gybp[-1] ));

        if( fabs(gx) < fabs(gy) )
            (q   = fabs(gx) / fabs(gy),
             gm2 = sqrtf( gxtp[0] * gxtp[0] + gytp[0] * gytp[0] ),
             gm4 = sqrtf( gxbp[0] * gxbp[0] + gybp[0] * gybp[0] ));
        else
            (q   = fabs(gy) / fabs(gx),
             gm2 = sqrtf( gxmp[ 1] * gxmp[ 1] + gymp[ 1] * gymp[ 1] ),
             gm4 = sqrtf( gxmp[-1] * gxmp[-1] + gymp[-1] * gymp[-1] ));

        if( gm < (gm2 + q * (gm1 - gm2)) ||
            gm < (gm4 + q * (gm3 - gm4)) )
            gm = 0.0f;
    }

    *dstp = gm;
}

/*************************************************************************/
/* CudaNonmaxSuppressionFilter                                           */
/*************************************************************************/
CudaNonmaxSuppressionFilter::CudaNonmaxSuppressionFilter(
    IImageFilter* first,
    IImageFilter* second
    )
: INonmaxSuppressionFilter( first, second )
{
}

void
CudaNonmaxSuppressionFilter::nonmaxSuppress(
    IImage& dest,
    const IImage& first,
    const IImage& second
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (dest.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (dest.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    nonmaxSuppressKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        first.data(), first.stride(),
        second.data(), second.stride(),
        dest.rows(), dest.columns()
        );

    cudaCheckLastError( "CudaNonmaxSuppressionFilter: kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaNonmaxSuppressionFilter: kernel run failed" );
}
