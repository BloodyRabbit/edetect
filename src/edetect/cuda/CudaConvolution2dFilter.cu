/** @file
 * @brief Definition of CudaConvolution2dFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cuda/CudaConvolution2dFilter.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaImage.hxx"

/// The convolution kernel.
__constant__ float cKernel[(2 * CudaConvolution2dFilter::MAX_RADIUS + 1) * (2 * CudaConvolution2dFilter::MAX_RADIUS + 1)];

/**
 * @brief CUDA kernel performing 2D discrete convolution.
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
convolve2d(
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
        const size_t colStart = (col < r
                                 ? r - col : 0);
        const size_t colEnd = (cols <= col + r
                               ? cols - col + r - 1
                               : 2 * r);

        const size_t rowStart = (row < r
                                 ? r - row : 0);
        const size_t rowEnd = (rows <= row + r
                               ? rows - row + r - 1
                               : 2 * r);

        const unsigned char* rowp = src
            + (row - r + rowStart) * srcStride
            + (col - r + colStart) * sizeof(float);
        const float* colp;

        size_t i, j;
        float x = 0.0f;

        for( i = 0; i < rowStart; ++i )
        {
            colp = (float*)rowp;
            for( j = 0; j < colStart; ++j )
                x += *colp * cKernel[i * (2 * r + 1) + j];
            for(; j < colEnd; ++j, ++colp )
                x += *colp * cKernel[i * (2 * r + 1) + j];
            for(; j <= 2 * r; ++j )
                x += *colp * cKernel[i * (2 * r + 1) + j];
        }
        for(; i < rowEnd; ++i, rowp += srcStride )
        {
            colp = (float*)rowp;
            for( j = 0; j < colStart; ++j )
                x += *colp * cKernel[i * (2 * r + 1) + j];
            for(; j < colEnd; ++j, ++colp )
                x += *colp * cKernel[i * (2 * r + 1) + j];
            for(; j <= 2 * r; ++j )
                x += *colp * cKernel[i * (2 * r + 1) + j];
        }
        for(; i <= 2 * r; ++i )
        {
            colp = (float*)rowp;
            for( j = 0; j < colStart; ++j )
                x += *colp * cKernel[i * (2 * r + 1) + j];
            for(; j < colEnd; ++j, ++colp )
                x += *colp * cKernel[i * (2 * r + 1) + j];
            for(; j <= 2 * r; ++j )
                x += *colp * cKernel[i * (2 * r + 1) + j];
        }

        float* dstp = (float*)(dst + row * dstStride)
            + col;
        *dstp = x;
    }
}

/*************************************************************************/
/* CudaConvolution2dFilter                                               */
/*************************************************************************/
CudaConvolution2dFilter::CudaConvolution2dFilter()
: mKernel( NULL ),
  mRadius( 0 )
{
}

void
CudaConvolution2dFilter::filter(
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
            "CudaConvolution2dFilter: Unsupported image format" );
    }

    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (image.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (image.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    CudaImage newImage;
    newImage.reset( image.rows(), image.columns(),
                    Image::FMT_GRAY_FLOAT32 );

    cudaCheckError(
        cudaMemcpyToSymbol(
            cKernel, mKernel, (2 * mRadius + 1) * (2 * mRadius + 1) * sizeof(*mKernel),
            0, cudaMemcpyHostToDevice ) );

    convolve2d<<< numBlocks, threadsPerBlock >>>(
        newImage.data(), newImage.stride(),
        image.data(), image.stride(),
        image.rows(), image.columns(), mRadius );

    cudaCheckLastError( "2D-convolution kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "2D-convolution kernel run failed" );

    image.swap( newImage );
}

void
CudaConvolution2dFilter::setParam(
    const char* name,
    const void* value
    )
{
    if( !strcmp( name, "kernel" ) )
        setKernel( (const float*)value, mRadius );
    else if( !strcmp( name, "radius" ) )
        setKernel( mKernel, *(const unsigned int*)value );
    else
        IImageFilter::setParam( name, value );
}

void
CudaConvolution2dFilter::setKernel(
    const float* kernel,
    unsigned int radius
    )
{
    if( MAX_RADIUS < radius )
        throw std::invalid_argument(
            "CudaConvolution2dFilter: Convolution kernel too large" );

    mKernel = kernel;
    mRadius = radius;
}
