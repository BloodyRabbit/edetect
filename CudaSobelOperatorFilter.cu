/** @file
 * @brief Definitio of CudaSobelOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "common.hxx"
#include "CudaError.hxx"
#include "CudaImage.hxx"
#include "CudaSobelOperatorFilter.hxx"

/**
 * @brief CUDA kernel which computes final gradient
 *   as given by Sobel operator.
 *
 * @param[in,out] vertEdges
 *   Detected vertical edges.
 * @param[in] vertStride
 *   Stride in vertical edges data.
 * @param[in] horzEdges
 *   Detected horizontal edges.
 * @param[in] horzStride
 *   Stride in horizontal edges data.
 * @param[in] rows
 *   Number of rows in data.
 * @param[in] cols
 *   Number of columns in data.
 */
__global__ void
computeGradientSobel(
    unsigned char* vertEdges,
    unsigned int vertStride,
    const unsigned char* horzEdges,
    unsigned int horzStride,
    unsigned int rows,
    unsigned int cols
    )
{
    const size_t col =
        blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if( row < rows && col < cols )
    {
        float* const vertp =
            (float*)(vertEdges + row * vertStride) + col;
        const float* const horzp =
            (const float*)(horzEdges + row * horzStride) + col;

        *vertp = sqrtf( (*vertp) * (*vertp) + (*horzp) * (*horzp) );
    }
}

/*************************************************************************/
/* CudaSobelOperatorFilter                                               */
/*************************************************************************/
const float
CudaSobelOperatorFilter::KERNEL_1_0_1[] =
{
    -1.0f, 0.0f, 1.0f
};

const float
CudaSobelOperatorFilter::KERNEL_1_2_1[] =
{
    1.0f, 2.0f, 1.0f
};

CudaSobelOperatorFilter::CudaSobelOperatorFilter()
: mVertFilter(
    KERNEL_1_0_1, KERNEL_RADIUS,
    KERNEL_1_2_1, KERNEL_RADIUS ),
  mHorzFilter(
    KERNEL_1_2_1, KERNEL_RADIUS,
    KERNEL_1_0_1, KERNEL_RADIUS )
{
}

void
CudaSobelOperatorFilter::process(
    CudaImage& image
    )
{
    switch( image.format() )
    {
    case CudaImage::FMT_GRAY_FLOAT32:
        break;

    default:
    case CudaImage::FMT_GRAY_UINT8:
    case CudaImage::FMT_RGB_UINT8:
    case CudaImage::FMT_RGB_FLOAT32:
        throw std::runtime_error(
            "CudaSobelOperatorFilter: Unsupported image format" );
    }

    CudaImage dupImage( image );

    // Detect vertical edges
    mVertFilter.process( image );
    // Detect horizontal edges
    mHorzFilter.process( dupImage );

    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (image.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (image.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    computeGradientSobel<<< numBlocks, threadsPerBlock >>>(
        (unsigned char*)image.data(), image.rowStride(),
        (unsigned char*)dupImage.data(), dupImage.rowStride(),
        image.rows(), image.columns() );

    cudaCheckLastError( "Sobel gradient computation kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "Sobel gradient computation kernel run failed" );
}
