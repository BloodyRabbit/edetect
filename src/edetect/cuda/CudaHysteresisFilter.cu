/** @file
 * @brief Definition of class CudaHysteresisFilter.
 *
 * @author Jan Bobek
 * @since 26th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaHysteresisFilter.hxx"
#include "cuda/CudaRingBuffer.cuh"

// A helper typedef for improved readability.
typedef CudaRingBuffer< ushort2 > CudaQueue;

/**
 * @brief CUDA kernel which performs
 *   initialization hysteresis step.
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
 * @param[in] thresh
 *   The high threshold.
 * @param[in] queue
 *   The queue to use.
 */
__global__ void
hysteresisInitKernel(
    unsigned char* ddata,
    unsigned int dstride,
    const unsigned char* sdata,
    unsigned int sstride,
    unsigned int rows,
    unsigned int cols,
    double thresh,
    CudaQueue& queue
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

    if( *srcp < thresh )
        *dstp = 0.0f;
    else
    {
        *dstp = 1.0f;
        queue.push_back( make_ushort2(row, col) );
    }
}

/**
 * @brief Helper function which attempts
 *   to enqueue a point.
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
 * @param[in] thresh
 *   The low threshold.
 * @param[in] queue
 *   The queue to use.
 * @param[in] pt
 *   The point to enqueue.
 */
__device__ void
hysteresisSearchEnqueue(
    unsigned char* ddata,
    unsigned int dstride,
    const unsigned char* sdata,
    unsigned int sstride,
    unsigned int rows,
    unsigned int cols,
    double thresh,
    CudaQueue& queue,
    ushort2 pt
    )
{
    if( !(pt.x < rows && pt.y < cols) )
        return;

    float* const dstp =
        (float*)(ddata + pt.x * dstride) + pt.y;
    const float* const srcp =
        (const float*)(sdata + pt.x * sstride) + pt.y;

    if( *srcp < thresh )
        return;

    /* This fence is required so that we get
       fresh value of *dstp in the condition below
       and avoid enqueueing a point twice or more. */
    __threadfence();
    if( 1.0f == *dstp )
        return;

    *dstp = 1.0f;
    queue.push_back( pt );
}

/**
 * @brief CUDA kernel which performs a search
 *   for edge points using a queue.
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
 * @param[in] thresh
 *   The low threshold.
 * @param[in] queue
 *   The queue to use.
 */
__global__ void
hysteresisSearchKernel(
    unsigned char* ddata,
    unsigned int dstride,
    const unsigned char* sdata,
    unsigned int sstride,
    unsigned int rows,
    unsigned int cols,
    double thresh,
    CudaQueue& queue
    )
{
    bool nonempty;
    ushort2 pt;

    while( __any( (nonempty = queue.pop_front( pt )) ) )
    {
        if( !nonempty )
            continue;

        hysteresisSearchEnqueue(
            ddata, dstride, sdata, sstride,
            rows, cols, thresh, queue,
            make_ushort2(pt.x - 1, pt.y - 1)
            );
        hysteresisSearchEnqueue(
            ddata, dstride, sdata, sstride,
            rows, cols, thresh, queue,
            make_ushort2(pt.x - 1, pt.y)
            );
        hysteresisSearchEnqueue(
            ddata, dstride, sdata, sstride,
            rows, cols, thresh, queue,
            make_ushort2(pt.x - 1, pt.y + 1)
            );

        hysteresisSearchEnqueue(
            ddata, dstride, sdata, sstride,
            rows, cols, thresh, queue,
            make_ushort2(pt.x, pt.y - 1)
            );
        hysteresisSearchEnqueue(
            ddata, dstride, sdata, sstride,
            rows, cols, thresh, queue,
            make_ushort2(pt.x, pt.y + 1)
            );

        hysteresisSearchEnqueue(
            ddata, dstride, sdata, sstride,
            rows, cols, thresh, queue,
            make_ushort2(pt.x + 1, pt.y - 1)
            );
        hysteresisSearchEnqueue(
            ddata, dstride, sdata, sstride,
            rows, cols, thresh, queue,
            make_ushort2(pt.x + 1, pt.y)
            );
        hysteresisSearchEnqueue(
            ddata, dstride, sdata, sstride,
            rows, cols, thresh, queue,
            make_ushort2(pt.x + 1, pt.y + 1)
            );
    }
}

/*************************************************************************/
/* CudaHysteresisFilter                                                  */
/*************************************************************************/
void
CudaHysteresisFilter::hysteresis(
    IImage& dest,
    const IImage& src
    )
{
    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (src.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (src.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    CudaQueue* const queue =
        CudaQueue::create( src.rows() * src.columns() );

    hysteresisInitKernel<<< numBlocks, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns(),
        mThresholdHigh, *queue
        );

    cudaCheckLastError( "CudaHysteresisFilter: initialization kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaHysteresisFilter: initialization kernel run failed" );

    hysteresisSearchKernel<<< 4, threadsPerBlock >>>(
        dest.data(), dest.stride(),
        src.data(), src.stride(),
        src.rows(), src.columns(),
        mThresholdLow, *queue
        );

    cudaCheckLastError( "CudaHysteresisFilter: search kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "CudaHysteresisFilter: search kernel run failed" );

    CudaQueue::destroy( queue );
}
