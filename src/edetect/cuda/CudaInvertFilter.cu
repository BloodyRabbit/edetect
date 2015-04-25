/** @file
 * @brief Definition of class CudaInvertFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaInvertFilter.hxx"

/**
 * @brief CUDA kernel which performs inversion.
 *
 * @param[in,out] data
 *   The image data.
 * @param[in] stride
 *   Size of the row stride in data.
 * @param[in] rows
 *   Number of rows in the image.
 * @param[in] cols
 *   Number of columns in the image.
 */
__global__ void
invertKernel(
    unsigned char* data,
    unsigned int stride,
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

    float* const dp =
        (float*)(data + row * stride) + col;

    *dp = 1.0f - *dp;
}

/*************************************************************************/
/* CudaInvertFilter                                                      */
/*************************************************************************/
void
CudaInvertFilter::invert(
    IImage& image
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (image.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (image.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    invertKernel<<< numBlocks, threadsPerBlock >>>(
        image.data(), image.stride(),
        image.rows(), image.columns()
        );

    cudaCheckLastError( "CudaInvertFilter: kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaInvertFilter: kernel run failed" );
}
