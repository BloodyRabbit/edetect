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
CudaConvolution2dSeparableFilter::CudaConvolution2dSeparableFilter()
: mRowKernel( NULL ),
  mRowKernelRadius( 0 ),
  mColumnKernel( NULL ),
  mColumnKernelRadius( 0 )
{
}

void
CudaConvolution2dSeparableFilter::filter(
    CudaImage& image
    )
{
    switch( image.format() )
    {
    case Image::FMT_GRAY_FLOAT32:
        break;

    default:
    case Image::FMT_GRAY_UINT8:
    case Image::FMT_RGB_UINT8:
    case Image::FMT_RGB_FLOAT32:
        throw std::runtime_error(
            "CudaConvolution2dSeparableFilter: Unsupported image format" );
    }

    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (image.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (image.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    CudaImage tempImage;
    tempImage.reset( image.rows(), image.columns(),
                     Image::FMT_GRAY_FLOAT32 );

    cudaCheckError(
        cudaMemcpyToSymbol(
            cKernel, mRowKernel, (2 * mRowKernelRadius + 1) * sizeof(*mRowKernel),
            0, cudaMemcpyHostToDevice ) );

    convolve1dRows<<< numBlocks, threadsPerBlock >>>(
        tempImage.data(), tempImage.stride(),
        image.data(), image.stride(),
        image.rows(), image.columns(), mRowKernelRadius );

    cudaCheckLastError( "2D-separable-convolution row kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "2D-separable-convolution row kernel run failed" );

    cudaCheckError(
        cudaMemcpyToSymbol(
            cKernel, mColumnKernel, (2 * mColumnKernelRadius + 1) * sizeof(*mColumnKernel),
            0, cudaMemcpyHostToDevice ) );

    convolve1dColumns<<< numBlocks, threadsPerBlock >>>(
        image.data(), image.stride(),
        tempImage.data(), tempImage.stride(),
        tempImage.rows(), tempImage.columns(), mColumnKernelRadius );

    cudaCheckLastError( "2D-separable-convolution column kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "2D-separable-convolution column kernel run failed" );
}

void
CudaConvolution2dSeparableFilter::setParam(
    const char* name,
    const void* value
    )
{
    if( !strcmp( name, "row-kernel" ) )
        setRowKernel( (const float*)value, mRowKernelRadius );
    else if( !strcmp( name, "row-kernel-radius" ) )
        setRowKernel( mRowKernel, *(const unsigned int*)value );
    else if( !strcmp( name, "column-kernel" ) )
        setColumnKernel( (const float*)value, mColumnKernelRadius );
    else if( !strcmp( name, "column-kernel-radius" ) )
        setColumnKernel( mColumnKernel, *(const unsigned int*)value );
    else
        IImageFilter::setParam( name, value );
}

void
CudaConvolution2dSeparableFilter::setRowKernel(
    const float* kernel,
    unsigned int radius
    )
{
    if( MAX_RADIUS < radius )
        throw std::invalid_argument(
            "CudaConvolution2dSeparableFilter: Row kernel radius too large" );

    mRowKernel = kernel;
    mRowKernelRadius = radius;
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

    mColumnKernel = kernel;
    mColumnKernelRadius = radius;
}
