/** @file
 * @brief Definition of CudaSobelOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaSobelOperatorFilter.hxx"

/**
 * @brief CUDA kernel which computes final gradient
 *   as given by Sobel operator.
 *
 * @param[in,out] vdata
 *   Detected vertical edges.
 * @param[in] vstride
 *   Stride in vertical edges data.
 * @param[in] hdata
 *   Detected horizontal edges.
 * @param[in] hstride
 *   Stride in horizontal edges data.
 * @param[in] rows
 *   Number of rows in data.
 * @param[in] cols
 *   Number of columns in data.
 */
__global__ void
computeGradientSobelKernel(
    unsigned char* vdata,
    unsigned int vstride,
    const unsigned char* hdata,
    unsigned int hstride,
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
        float* const vertp =
            (float*)(vdata + row * vstride) + col;
        const float* const horzp =
            (const float*)(hdata + row * hstride) + col;

        *vertp = sqrtf( (*vertp) * (*vertp) + (*horzp) * (*horzp) );
    }
}

/*************************************************************************/
/* CudaSobelOperatorFilter                                               */
/*************************************************************************/
void
CudaSobelOperatorFilter::computeGradient(
    IImage& vert,
    const IImage& horz
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (vert.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (vert.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    computeGradientSobelKernel<<< numBlocks, threadsPerBlock >>>(
        vert.data(), vert.stride(),
        horz.data(), horz.stride(),
        vert.rows(), vert.columns()
        );

    cudaCheckLastError( "CudaSobelOperatorFilter: gradient computation kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaSobelOperatorFilter: gradient computation kernel run failed" );
}
