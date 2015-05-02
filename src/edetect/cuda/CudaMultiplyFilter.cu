/** @file
 * @brief Definition of class CudaMultiplyFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaMultiplyFilter.hxx"

/**
 * @brief CUDA kernel which multiplies one
 *   image with another.
 *
 * @param[in,out] ddata
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
multiplyKernel(
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
    const float* const srcp =
        (const float*)(sdata + row * sstride) + col;

    *dstp *= *srcp;
}

/*************************************************************************/
/* CudaMultiplyFilter                                                    */
/*************************************************************************/
CudaMultiplyFilter::CudaMultiplyFilter(
    IImageFilter* first,
    IImageFilter* second
    )
: IMultiplyFilter( first, second )
{
}

void
CudaMultiplyFilter::filter2(
    IImage& first,
    const IImage& second
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (first.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (first.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    multiplyKernel<<< numBlocks, threadsPerBlock >>>(
        first.data(), first.stride(),
        second.data(), second.stride(),
        first.rows(), first.columns()
        );

    cudaCheckLastError( "CudaMultiplyFilter: kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaMultiplyFilter: kernel run failed" );
}
