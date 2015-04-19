/** @file
 * @brief Definition of class CudaEuclideanNormFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaEuclideanNormFilter.hxx"

/**
 * @brief CUDA kernel which computes the Euclidean norm.
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
computeEuclideanNormKernel(
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
        const float* const srcp =
            (const float*)(sdata + row * sstride) + col;

        *dstp = sqrtf( (*srcp) * (*srcp) + (*dstp) * (*dstp) );
    }
}

/*************************************************************************/
/* CudaEuclideanNormFilter                                               */
/*************************************************************************/
CudaEuclideanNormFilter::CudaEuclideanNormFilter(
    IImageFilter* first,
    IImageFilter* second
    )
: IEuclideanNormFilter( first, second )
{
}

void
CudaEuclideanNormFilter::filter2(
    IImage& first,
    const IImage& second
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (first.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (first.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    computeEuclideanNormKernel<<< numBlocks, threadsPerBlock >>>(
        first.data(), first.stride(),
        second.data(), second.stride(),
        first.rows(), first.columns()
        );

    cudaCheckLastError( "CudaEuclideanNormFilter: kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaEuclideanNormFilter: kernel run failed" );
}
