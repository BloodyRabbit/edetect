/** @file
 * @brief Definition of CudaConvolution2dSeparableFilter class.
 *
 * @author Jan Bobek
 */

#include "common.hxx"
#include "cuda/CudaConvolution2dSeparableFilter.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaImage.hxx"

/// The convolution kernel.
__constant__ float cKernel[2 * CudaConvolution2dSeparableFilter::MAX_RADIUS + 1];

/**
 * @brief CUDA kernel for one-dimensional
 *   convolution along rows.
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
 * @param[in] r
 *   Radius of the kernel.
 */
__global__ void
convolve1dRows(
    unsigned char* dst,
    unsigned int dstStride,
    const unsigned char* src,
    unsigned int srcStride,
    unsigned int rows,
    unsigned int cols,
    unsigned int r
    )
{
    const size_t col =
        blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if( row < rows && col < cols )
    {
        const size_t start = (col < r
                              ? r - col : 0);
        const size_t end = (cols <= col + r
                            ? cols - col + r - 1
                            : 2 * r);

        const float* srcp =
            (const float*)(src + row * srcStride)
            + (col - r + start);

        size_t i;
        float x = 0.0f;

        for( i = 0; i < start; ++i )
            x += *srcp * cKernel[i];
        for(; i < end; ++i, ++srcp )
            x += *srcp * cKernel[i];
        for(; i <= 2 * r; ++i )
            x += *srcp * cKernel[i];

        float* dstp = (float*)(dst + row * dstStride)
            + col;
        *dstp = x;
    }
}

/**
 * @brief CUDA kernel for one-dimensional
 *   convolution along columns.
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
 * @param[in] r
 *   Radius of the kernel.
 */
__global__ void
convolve1dColumns(
    unsigned char* dst,
    unsigned int dstStride,
    const unsigned char* src,
    unsigned int srcStride,
    unsigned int rows,
    unsigned int cols,
    unsigned int r
    )
{
    const size_t col =
        blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if( row < rows && col < cols )
    {
        const size_t start = (row < r
                              ? r - row : 0);
        const size_t end = (rows <= row + r
                            ? rows - row + r - 1
                            : 2 * r);

        const unsigned char* srcp = src
            + (row - r + start) * srcStride
            + col * sizeof(float);

        size_t i;
        float x = 0.0f;

        for( i = 0; i < start; ++i )
            x += *(float*)srcp * cKernel[i];
        for(; i < end; ++i, srcp += srcStride )
            x += *(float*)srcp * cKernel[i];
        for(; i <= 2 * r; ++i )
            x += *(float*)srcp * cKernel[i];

        float* dstp = (float*)(dst + row * dstStride)
            + col;
        *dstp = x;
    }
}

/*************************************************************************/
/* CudaConvolution2dSeparableFilter                                      */
/*************************************************************************/
CudaConvolution2dSeparableFilter::CudaConvolution2dSeparableFilter(
    const float* kernelRows,
    unsigned int kernelRowsRadius,
    const float* kernelColumns,
    unsigned int kernelColumnsRadius
    )
: mKernelRows( NULL ),
  mKernelRowsRadius( 0 ),
  mKernelColumns( NULL ),
  mKernelColumnsRadius( 0 )
{
    setKernelRows( kernelRows, kernelRowsRadius );
    setKernelColumns( kernelColumns, kernelColumnsRadius );
}

void
CudaConvolution2dSeparableFilter::process(
    CudaImage& image
    )
{
    switch( image.format() )
    {
    case CudaImage::FMT_GRAY_FLOAT32:
        break;

    default:
    case CudaImage::FMT_GRAY_UINT8:
    case CudaImage::FMT_RGB_UINT8:
    case CudaImage::FMT_RGB_FLOAT32:
        throw std::runtime_error(
            "CudaConvolution2dSeparableFilter: Unsupported image format" );
    }

    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (image.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (image.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    CudaImage tempImage(
        image.rows(), image.columns(),
        CudaImage::FMT_GRAY_FLOAT32 );

    cudaCheckError(
        cudaMemcpyToSymbol(
            cKernel, mKernelRows, (2 * mKernelRowsRadius + 1) * sizeof(*mKernelRows),
            0, cudaMemcpyHostToDevice ) );

    convolve1dRows<<< numBlocks, threadsPerBlock >>>(
        (unsigned char*)tempImage.data(), tempImage.rowStride(),
        (unsigned char*)image.data(), image.rowStride(),
        image.rows(), image.columns(), mKernelRowsRadius );

    cudaCheckLastError( "2D-separable-convolution row kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "2D-separable-convolution row kernel run failed" );

    cudaCheckError(
        cudaMemcpyToSymbol(
            cKernel, mKernelColumns, (2 * mKernelColumnsRadius + 1) * sizeof(*mKernelColumns),
            0, cudaMemcpyHostToDevice ) );

    convolve1dColumns<<< numBlocks, threadsPerBlock >>>(
        (unsigned char*)image.data(), image.rowStride(),
        (unsigned char*)tempImage.data(), tempImage.rowStride(),
        tempImage.rows(), tempImage.columns(), mKernelColumnsRadius );

    cudaCheckLastError( "2D-separable-convolution column kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "2D-separable-convolution column kernel run failed" );
}

void
CudaConvolution2dSeparableFilter::setKernelRows(
    const float* kernel,
    unsigned int radius
    )
{
    if( MAX_RADIUS < radius )
        throw std::runtime_error(
            "CudaConvolution2dSeparableFilter: Row kernel radius too large" );

    mKernelRows = kernel;
    mKernelRowsRadius = radius;
}

void
CudaConvolution2dSeparableFilter::setKernelColumns(
    const float* kernel,
    unsigned int radius
    )
{
    if( MAX_RADIUS < radius )
        throw std::runtime_error(
            "CudaConvolution2dSeparableFilter: Column kernel radius too large" );

    mKernelColumns = kernel;
    mKernelColumnsRadius = radius;
}
