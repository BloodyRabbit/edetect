/** @file
 * @brief Definition of CudaConvolutionFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaConvolutionFilter.hxx"
#include "cuda/CudaError.hxx"

/// The convolution kernel.
__constant__ float cKernel[ICudaConvolutionFilter::MAX_LENGTH];

/**
 * @brief CUDA kernel performing 2D discrete convolution.
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
convolveKernel(
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

    if( !(row < rows && col < cols) )
        return;

    const unsigned int cstart =
        (col < r ? r - col : 0);
    const unsigned int rstart =
        (row < r ? r - row : 0);

    const unsigned int cend =
        (cols <= col + r ? cols - col + r - 1 : 2 * r);
    const unsigned int rend =
        (rows <= row + r ? rows - row + r - 1 : 2 * r);

    float* dstp =
        (float*)(ddata + row * dstride)
        + col;
    const unsigned char* rowp = sdata
        + (row - r + rstart) * sstride
        + (col - r + cstart) * sizeof(float);
    const float* colp;

    unsigned int i, j;
    float x = 0.0f;

    for( i = 0; i < rstart; ++i )
    {
        colp = (float*)rowp;
        for( j = 0; j < cstart; ++j )
            x += *colp * cKernel[i * (2 * r + 1) + j];
        for(; j < cend; ++j, ++colp )
            x += *colp * cKernel[i * (2 * r + 1) + j];
        for(; j <= 2 * r; ++j )
            x += *colp * cKernel[i * (2 * r + 1) + j];
    }
    for(; i < rend; ++i, rowp += sstride )
    {
        colp = (float*)rowp;
        for( j = 0; j < cstart; ++j )
            x += *colp * cKernel[i * (2 * r + 1) + j];
        for(; j < cend; ++j, ++colp )
            x += *colp * cKernel[i * (2 * r + 1) + j];
        for(; j <= 2 * r; ++j )
            x += *colp * cKernel[i * (2 * r + 1) + j];
    }
    for(; i <= 2 * r; ++i )
    {
        colp = (float*)rowp;
        for( j = 0; j < cstart; ++j )
            x += *colp * cKernel[i * (2 * r + 1) + j];
        for(; j < cend; ++j, ++colp )
            x += *colp * cKernel[i * (2 * r + 1) + j];
        for(; j <= 2 * r; ++j )
            x += *colp * cKernel[i * (2 * r + 1) + j];
    }

    *dstp = x;
}

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
convolveRowsKernel(
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

    if( !(row < rows && col < cols) )
        return;

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
convolveColumnsKernel(
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

    if( !(row < rows && col < cols) )
        return;

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

/*************************************************************************/
/* ICudaConvolutionFilter                                                */
/*************************************************************************/
void
ICudaConvolutionFilter::setKernel(
    const float* kernel,
    unsigned int length
    )
{
    if( MAX_LENGTH < length )
        throw std::invalid_argument(
            "ICudaConvolutionFilter: Convolution kernel too large" );

    IConvolutionFilter::setKernel( kernel, length );
}

/*************************************************************************/
/* CudaConvolutionFilter                                                 */
/*************************************************************************/
void
CudaConvolutionFilter::convolve(
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
            cKernel, mKernel, mLength * sizeof(*mKernel),
            0, cudaMemcpyHostToDevice ) );

    convolveKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns(),
        ((unsigned int)std::sqrt( mLength ) - 1) / 2
        );

    cudaCheckLastError( "CudaConvolutionFilter: kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaConvolutionFilter: kernel run failed" );
}

/*************************************************************************/
/* CudaRowConvolutionFilter                                              */
/*************************************************************************/
void
CudaRowConvolutionFilter::convolve(
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
            cKernel, mKernel, mLength * sizeof(*mKernel),
            0, cudaMemcpyHostToDevice ) );

    convolveRowsKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns(),
        (mLength - 1) / 2
        );

    cudaCheckLastError( "CudaRowConvolutionFilter: kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaRowConvolutionFilter: kernel run failed" );
}

/*************************************************************************/
/* CudaColumnConvolutionFilter                                           */
/*************************************************************************/
void
CudaColumnConvolutionFilter::convolve(
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
            cKernel, mKernel, mLength * sizeof(*mKernel),
            0, cudaMemcpyHostToDevice ) );

    convolveColumnsKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns(),
        (mLength - 1) / 2
        );

    cudaCheckLastError( "CudaColumnConvolutionFilter: column kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaColumnConvolutionFilter: column kernel run failed" );
}
