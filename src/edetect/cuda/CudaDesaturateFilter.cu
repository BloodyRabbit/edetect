/** @file
 * @brief Definition of CudaDesaturateFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaDesaturateFilter.hxx"
#include "cuda/CudaError.hxx"

/**
 * @brief CUDA kernel for desaturation using
 *   the Average method.
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
desaturateAverageKernel(
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

    if( row < rows && col < cols )
    {
        float* const dstp =
            (float*)(ddata + row * dstride) + col;
        const float3* const srcp =
            (const float3*)(sdata + row * sstride) + col;

        *dstp = (srcp->x + srcp->y + srcp->z) / 3.0f;
    }
}

/**
 * @brief CUDA kernel for desaturation using
 *   the Lightness method.
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
desaturateLightnessKernel(
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

    if( row < rows && col < cols )
    {
        float* const dstp =
            (float*)(ddata + row * dstride) + col;
        const float3* const srcp =
            (const float3*)(sdata + row * sstride) + col;

        const float a = fminf( srcp->x, srcp->y );
        const float b = fmaxf( srcp->x, srcp->y );
        const float c = fminf( a, srcp->z );
        const float d = fmaxf( b, srcp->z );

        *dstp = 0.5f * (c + d);
    }
}

/**
 * @brief CUDA kernel for desaturation using
 *   the Luminosity method.
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
desaturateLuminosityKernel(
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

    if( row < rows && col < cols )
    {
        float* const dstp =
            (float*)(ddata + row * dstride) + col;
        const float3* const srcp =
            (const float3*)(sdata + row * sstride) + col;

        *dstp =
            /* z:RED y:GREEN x:BLUE */
            0.2126f * srcp->z +
            0.7152f * srcp->y +
            0.0722f * srcp->x;
    }
}

/*************************************************************************/
/* CudaDesaturateFilter                                                  */
/*************************************************************************/
void
CudaDesaturateFilter::desaturateAverage(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    desaturateAverageKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaDesaturateFilter: Average kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaDesaturateFilter: Average kernel run failed" );
}

void
CudaDesaturateFilter::desaturateLightness(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    desaturateLightnessKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaDesaturateFilter: Lightness kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaDesaturateFilter: Lightness kernel run failed" );
}

void
CudaDesaturateFilter::desaturateLuminosity(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    desaturateLuminosityKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaDesaturateFilter: Luminosity kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaDesaturateFilter: Luminosity kernel run failed" );
}
