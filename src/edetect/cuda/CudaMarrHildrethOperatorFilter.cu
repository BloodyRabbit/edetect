/** @file
 * @brief Definition of CudaMarrHildrethOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaMarrHildrethOperatorFilter.hxx"

/**
 * @brief CUDA kernel which merges edge pixels
 *   in supplied images.
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
mergeEdgesMarrHildrethKernel(
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

        *dstp *= *srcp;
    }
}

/*************************************************************************/
/* CudaMarrHildrethOperatorFilter                                        */
/*************************************************************************/
void
CudaMarrHildrethOperatorFilter::mergeEdges(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    mergeEdgesMarrHildrethKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns()
        );

    cudaCheckLastError( "CudaMarrHildrethOperatorFilter: edge merge kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaMarrHildrethOperatorFilter: edge merge kernel run failed" );
}
