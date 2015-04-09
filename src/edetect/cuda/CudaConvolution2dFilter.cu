/** @file
 * @brief Definition of CudaConvolution2dFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaConvolution2dFilter.hxx"
#include "cuda/CudaError.hxx"

/// The convolution kernel.
__constant__ float cKernel[(2 * CudaConvolution2dFilter::MAX_RADIUS + 1) * (2 * CudaConvolution2dFilter::MAX_RADIUS + 1)];

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
convolve2dKernel(
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
}

/*************************************************************************/
/* CudaConvolution2dFilter                                               */
/*************************************************************************/
void
CudaConvolution2dFilter::setKernel(
    const float* kernel,
    unsigned int radius
    )
{
    if( MAX_RADIUS < radius )
        throw std::invalid_argument(
            "CudaConvolution2dFilter: Convolution kernel too large" );

    IConvolution2dFilter::setKernel( kernel, radius );
}

void
CudaConvolution2dFilter::convolve(
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
            cKernel, mKernel, (2 * mRadius + 1) * (2 * mRadius + 1) * sizeof(*mKernel),
            0, cudaMemcpyHostToDevice ) );

    convolve2dKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns(),
        mRadius
        );

    cudaCheckLastError( "CudaConvolution2dFilter: kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaConvolution2dFilter: kernel run failed" );
}
