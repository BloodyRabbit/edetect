/** @file
 * @brief Definition of CudaConvolution2dSeparableFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cuda/CudaConvolution2dSeparableFilter.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaImage.hxx"

/// The convolution kernel.
__constant__ float cKernel[2 * CudaConvolution2dSeparableFilter::MAX_RADIUS + 1];

/**
 * @brief CUDA kernel for one-dimensional
 *   convolution along rows.
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
 * @param[in] r
 *   Radius of the kernel.
 */
__global__ void
convolve1dRowsKernel(
    unsigned char* ddata,
    unsigned int dstride,
    const unsigned char* sdata,
    unsigned int sstride,
    unsigned int rows,
    unsigned int cols,
    unsigned int r
    )
{
    const unsigned int col =
        blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if( row < rows && col < cols )
    {
        const unsigned int start =
            (col < r ? r - col : 0);
        const unsigned int end =
            (cols <= col + r
             ? cols - col + r - 1
             : 2 * r);

        float* dstp =
            (float*)(ddata + row * dstride)
            + col;
        const float* srcp =
            (const float*)(sdata + row * sstride)
            + (col - r + start);

        unsigned int i;
        float x = 0.0f;

        for( i = 0; i < start; ++i )
            x += *srcp * cKernel[i];
        for(; i < end; ++i, ++srcp )
            x += *srcp * cKernel[i];
        for(; i <= 2 * r; ++i )
            x += *srcp * cKernel[i];

        *dstp = x;
    }
}

/**
 * @brief CUDA kernel for one-dimensional
 *   convolution along columns.
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
 * @param[in] r
 *   Radius of the kernel.
 */
__global__ void
convolve1dColumnsKernel(
    unsigned char* ddata,
    unsigned int dstride,
    const unsigned char* sdata,
    unsigned int sstride,
    unsigned int rows,
    unsigned int cols,
    unsigned int r
    )
{
    const unsigned int col =
        blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if( row < rows && col < cols )
    {
        const unsigned int start =
            (row < r ? r - row : 0);
        const unsigned int end =
            (rows <= row + r
             ? rows - row + r - 1
             : 2 * r);

        float* dstp =
            (float*)(ddata + row * dstride)
            + col;
        const unsigned char* srcp = sdata
            + (row - r + start) * sstride
            + col * sizeof(float);

        unsigned int i;
        float x = 0.0f;

        for( i = 0; i < start; ++i )
            x += *(float*)srcp * cKernel[i];
        for(; i < end; ++i, srcp += sstride )
            x += *(float*)srcp * cKernel[i];
        for(; i <= 2 * r; ++i )
            x += *(float*)srcp * cKernel[i];

        *dstp = x;
    }
}

/*************************************************************************/
/* CudaConvolution2dSeparableFilter                                      */
/*************************************************************************/
void
CudaConvolution2dSeparableFilter::setRowKernel(
    const float* kernel,
    unsigned int radius
    )
{
    if( MAX_RADIUS < radius )
        throw std::invalid_argument(
            "CudaConvolution2dSeparableFilter: Row kernel radius too large" );

    IConvolution2dSeparableFilter::setRowKernel( kernel, radius );
}

void
CudaConvolution2dSeparableFilter::setColumnKernel(
    const float* kernel,
    unsigned int radius
    )
{
    if( MAX_RADIUS < radius )
        throw std::invalid_argument(
            "CudaConvolution2dSeparableFilter: Column kernel radius too large" );

    IConvolution2dSeparableFilter::setColumnKernel( kernel, radius );
}

void
CudaConvolution2dSeparableFilter::convolveRows(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    cudaCheckError(
        cudaMemcpyToSymbol(
            cKernel, mRowKernel, (2 * mRowKernelRadius + 1) * sizeof(*mRowKernel),
            0, cudaMemcpyHostToDevice ) );

    convolve1dRowsKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns(),
        mRowKernelRadius
        );

    cudaCheckLastError( "CudaConvolution2dSeparableFilter: row kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaConvolution2dSeparableFilter: row kernel run failed" );
}

void
CudaConvolution2dSeparableFilter::convolveColumns(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    cudaCheckError(
        cudaMemcpyToSymbol(
            cKernel, mColumnKernel, (2 * mColumnKernelRadius + 1) * sizeof(*mColumnKernel),
            0, cudaMemcpyHostToDevice ) );

    convolve1dColumnsKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns(),
        mColumnKernelRadius
        );

    cudaCheckLastError( "CudaConvolution2dSeparableFilter: column kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaConvolution2dSeparableFilter: column kernel run failed" );
}
