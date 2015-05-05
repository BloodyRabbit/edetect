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
 * @brief CUDA kernel for desaturation an integer
 *   image using the Average method.
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
desaturateAverageIntKernel(
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
    const uchar3* const srcp =
        (const uchar3*)(sdata + row * sstride) + col;

    *dstp = ((unsigned int)
             srcp->x + srcp->y + srcp->z) / 765.0f;
}

/**
 * @brief CUDA kernel for desaturation a floating-point
 *   image using the Average method.
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
desaturateAverageFloatKernel(
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
    const float3* const srcp =
        (const float3*)(sdata + row * sstride) + col;

    *dstp = (srcp->x + srcp->y + srcp->z) / 3.0f;
}

/**
 * @brief CUDA kernel for desaturation an integer
 *   image using the Lightness method.
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
desaturateLightnessIntKernel(
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
    const uchar3* const srcp =
        (const uchar3*)(sdata + row * sstride) + col;

    const unsigned char
        a = min( srcp->x, srcp->y ),
        b = max( srcp->x, srcp->y );
    const unsigned int
        c = min( a, srcp->z ),
        d = max( b, srcp->z );

    *dstp = (c + d) / 510.0f;
}

/**
 * @brief CUDA kernel for desaturation a floating-point
 *   image using the Lightness method.
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
desaturateLightnessFloatKernel(
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
    const float3* const srcp =
        (const float3*)(sdata + row * sstride) + col;

    const float
        a = fminf( srcp->x, srcp->y ),
        b = fmaxf( srcp->x, srcp->y ),
        c = fminf( a, srcp->z ),
        d = fmaxf( b, srcp->z );

    *dstp = 0.5f * (c + d);
}

/**
 * @brief CUDA kernel for desaturation an integer
 *   image using the Luminosity method.
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
desaturateLuminosityIntKernel(
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
    const uchar3* const srcp =
        (const uchar3*)(sdata + row * sstride) + col;

    *dstp =
        /* z:RED y:GREEN x:BLUE */
        0.2126f * srcp->z / 255.0f +
        0.7152f * srcp->y / 255.0f +
        0.0722f * srcp->x / 255.0f;
}

/**
 * @brief CUDA kernel for desaturation a floating-point
 *   image using the Luminosity method.
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
desaturateLuminosityFloatKernel(
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
    const float3* const srcp =
        (const float3*)(sdata + row * sstride) + col;

    *dstp =
        /* z:RED y:GREEN x:BLUE */
        0.2126f * srcp->z +
        0.7152f * srcp->y +
        0.0722f * srcp->x;
}

/*************************************************************************/
/* CudaDesaturateFilter                                                  */
/*************************************************************************/
void
CudaDesaturateFilter::desaturateAverageInt(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    desaturateAverageIntKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaDesaturateFilter: AverageInt kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaDesaturateFilter: AverageInt kernel run failed" );
}

void
CudaDesaturateFilter::desaturateAverageFloat(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    desaturateAverageFloatKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaDesaturateFilter: AverageFloat kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaDesaturateFilter: AverageFloat kernel run failed" );
}

void
CudaDesaturateFilter::desaturateLightnessInt(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    desaturateLightnessIntKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaDesaturateFilter: LightnessInt kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaDesaturateFilter: LightnessInt kernel run failed" );
}

void
CudaDesaturateFilter::desaturateLightnessFloat(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    desaturateLightnessFloatKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaDesaturateFilter: LightnessFloat kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaDesaturateFilter: LightnessFloat kernel run failed" );
}

void
CudaDesaturateFilter::desaturateLuminosityInt(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    desaturateLuminosityIntKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaDesaturateFilter: LuminosityInt kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaDesaturateFilter: LuminosityInt kernel run failed" );
}

void
CudaDesaturateFilter::desaturateLuminosityFloat(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    desaturateLuminosityFloatKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaDesaturateFilter: LuminosityFloat kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaDesaturateFilter: LuminosityFloat kernel run failed" );
}
