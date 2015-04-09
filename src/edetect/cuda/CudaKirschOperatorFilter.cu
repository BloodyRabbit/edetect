/** @file
 * @brief Definition of CudaKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaKirschOperatorFilter.hxx"

/**
 * @brief CUDA kernel computing Kirsch operator gradient.
 *
 * @param[in,out] sdata1
 *   The source image data #1.
 * @param[in] sstride1
 *   Size of the row stride in source data #1.
 * @param[in] sdata2
 *   The source image data #2.
 * @param[in] sstride2
 *   Size of the row stride in source data #2.
 * @param[in] sdata3
 *   The source image data #3.
 * @param[in] sstride3
 *   Size of the row stride in source data #3.
 * @param[in] sdata4
 *   The source image data #4.
 * @param[in] sstride4
 *   Size of the row stride in source data #4.
 * @param[in] sdata5
 *   The source image data #5.
 * @param[in] sstride5
 *   Size of the row stride in source data #5.
 * @param[in] sdata6
 *   The source image data #6.
 * @param[in] sstride6
 *   Size of the row stride in source data #6.
 * @param[in] sdata7
 *   The source image data #7.
 * @param[in] sstride7
 *   Size of the row stride in source data #7.
 * @param[in] sdata8
 *   The source image data #8.
 * @param[in] sstride8
 *   Size of the row stride in source data #8.
 * @param[in] rows
 *   Number of rows in the image.
 * @param[in] cols
 *   Number of columns in the image.
 */
__global__ void
computeGradientKirschKernel(
    unsigned char* sdata1,
    unsigned int sstride1,
    const unsigned char* sdata2,
    unsigned int sstride2,
    const unsigned char* sdata3,
    unsigned int sstride3,
    const unsigned char* sdata4,
    unsigned int sstride4,
    const unsigned char* sdata5,
    unsigned int sstride5,
    const unsigned char* sdata6,
    unsigned int sstride6,
    const unsigned char* sdata7,
    unsigned int sstride7,
    const unsigned char* sdata8,
    unsigned int sstride8,
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
        float* src1p =
            (float*)(sdata1 + row * sstride1) + col;

        const float* src2p =
            (const float*)(sdata2 + row * sstride2) + col;
        const float* src3p =
            (const float*)(sdata3 + row * sstride3) + col;
        const float* src4p =
            (const float*)(sdata4 + row * sstride4) + col;
        const float* src5p =
            (const float*)(sdata5 + row * sstride5) + col;
        const float* src6p =
            (const float*)(sdata6 + row * sstride6) + col;
        const float* src7p =
            (const float*)(sdata7 + row * sstride7) + col;
        const float* src8p =
            (const float*)(sdata8 + row * sstride8) + col;

        float x = fabs(*src1p);
        x = fmaxf( x, fabs(*src2p) );
        x = fmaxf( x, fabs(*src3p) );
        x = fmaxf( x, fabs(*src4p) );
        x = fmaxf( x, fabs(*src5p) );
        x = fmaxf( x, fabs(*src6p) );
        x = fmaxf( x, fabs(*src7p) );
        x = fmaxf( x, fabs(*src8p) );

        *src1p = x;
    }
}

/*************************************************************************/
/* CudaKirschOperatorFilter                                              */
/*************************************************************************/
void
CudaKirschOperatorFilter::computeGradient(
    IImage* images[KERNEL_COUNT]
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (images[0]->columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (images[0]->rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    computeGradientKirschKernel<<< numBlocks, threadsPerBlock >>>(
        images[0]->data(), images[0]->stride(),
        images[1]->data(), images[1]->stride(),
        images[2]->data(), images[2]->stride(),
        images[3]->data(), images[3]->stride(),
        images[4]->data(), images[4]->stride(),
        images[5]->data(), images[5]->stride(),
        images[6]->data(), images[6]->stride(),
        images[7]->data(), images[7]->stride(),
        images[0]->rows(), images[0]->columns()
        );

    cudaCheckLastError( "CudaKirschOperatorFilter: gradient computation kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaKirschOperatorFilter: gradient computation kernel run failed" );
}
