/** @file
 * @brief Definition of CudaKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaImage.hxx"
#include "cuda/CudaKirschOperatorFilter.hxx"

/**
 * @brief CUDA kernel computing Kirsch operator gradient.
 *
 * @param[in] src1
 *   The source image data #1.
 * @param[in] srcStride1
 *   Size of the row stride in source data #1.
 * @param[in] src2
 *   The source image data #2.
 * @param[in] srcStride2
 *   Size of the row stride in source data #2.
 * @param[in] src3
 *   The source image data #3.
 * @param[in] srcStride3
 *   Size of the row stride in source data #3.
 * @param[in] src4
 *   The source image data #4.
 * @param[in] srcStride4
 *   Size of the row stride in source data #4.
 * @param[in] src5
 *   The source image data #5.
 * @param[in] srcStride5
 *   Size of the row stride in source data #5.
 * @param[in] src6
 *   The source image data #6.
 * @param[in] srcStride6
 *   Size of the row stride in source data #6.
 * @param[in] src7
 *   The source image data #7.
 * @param[in] srcStride7
 *   Size of the row stride in source data #7.
 * @param[in] src8
 *   The source image data #8.
 * @param[in] srcStride8
 *   Size of the row stride in source data #8.
 * @param[in] rows
 *   Number of rows in the image.
 * @param[in] cols
 *   Number of columns in the image.
 */
__global__ void
computeGradientKirsch(
    unsigned char* src1,
    unsigned int srcStride1,
    const unsigned char* src2,
    unsigned int srcStride2,
    const unsigned char* src3,
    unsigned int srcStride3,
    const unsigned char* src4,
    unsigned int srcStride4,
    const unsigned char* src5,
    unsigned int srcStride5,
    const unsigned char* src6,
    unsigned int srcStride6,
    const unsigned char* src7,
    unsigned int srcStride7,
    const unsigned char* src8,
    unsigned int srcStride8,
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
        float* src1p =
            (float*)(src1 + row * srcStride1) + col;

        const float* src2p =
            (const float*)(src2 + row * srcStride2) + col;
        const float* src3p =
            (const float*)(src3 + row * srcStride3) + col;
        const float* src4p =
            (const float*)(src4 + row * srcStride4) + col;
        const float* src5p =
            (const float*)(src5 + row * srcStride5) + col;
        const float* src6p =
            (const float*)(src6 + row * srcStride6) + col;
        const float* src7p =
            (const float*)(src7 + row * srcStride7) + col;
        const float* src8p =
            (const float*)(src8 + row * srcStride8) + col;

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
const float
CudaKirschOperatorFilter::KERNELS[][(2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1)] =
{
    { -3.0f, -3.0f,  5.0f,
      -3.0f,  0.0f,  5.0f,
      -3.0f, -3.0f,  5.0f },

    { -3.0f,  5.0f,  5.0f,
      -3.0f,  0.0f,  5.0f,
      -3.0f, -3.0f, -3.0f },

    {  5.0f,  5.0f,  5.0f,
      -3.0f,  0.0f, -3.0f,
      -3.0f, -3.0f, -3.0f },

    {  5.0f,  5.0f, -3.0f,
       5.0f,  0.0f, -3.0f,
      -3.0f, -3.0f, -3.0f },

    {  5.0f, -3.0f, -3.0f,
       5.0f,  0.0f, -3.0f,
       5.0f, -3.0f, -3.0f },

    { -3.0f, -3.0f, -3.0f,
       5.0f,  0.0f, -3.0f,
       5.0f,  5.0f, -3.0f },

    { -3.0f, -3.0f, -3.0f,
      -3.0f,  0.0f, -3.0f,
       5.0f,  5.0f,  5.0f },

    { -3.0f, -3.0f, -3.0f,
      -3.0f,  0.0f,  5.0f,
      -3.0f,  5.0f,  5.0f },
};

CudaKirschOperatorFilter::CudaKirschOperatorFilter()
{
    for( unsigned int i = 0; i < KERNEL_COUNT; ++i )
        mFilters[i].setKernel( (const float*)&KERNELS[i], KERNEL_RADIUS );
}

void
CudaKirschOperatorFilter::process(
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
            "CudaKirschOperatorFilter: Unsupported image format" );
    }

    CudaImage* dupImages[KERNEL_COUNT] =
        { &image,
          new CudaImage( image ),
          new CudaImage( image ),
          new CudaImage( image ),

          new CudaImage( image ),
          new CudaImage( image ),
          new CudaImage( image ),
          new CudaImage( image ) };

    for( unsigned int i = 0; i < KERNEL_COUNT; ++i )
        mFilters[i].process( *dupImages[i] );

    // 32 = warp size, 8 * 32 = 256 threads
    const dim3 threadsPerBlock(32, 8);
    const dim3 numBlocks(
        (image.columns() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (image.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y );

    computeGradientKirsch<<< numBlocks, threadsPerBlock >>>(
        (unsigned char*)dupImages[0]->data(), dupImages[0]->rowStride(),
        (unsigned char*)dupImages[1]->data(), dupImages[1]->rowStride(),
        (unsigned char*)dupImages[2]->data(), dupImages[2]->rowStride(),
        (unsigned char*)dupImages[3]->data(), dupImages[3]->rowStride(),
        (unsigned char*)dupImages[4]->data(), dupImages[4]->rowStride(),
        (unsigned char*)dupImages[5]->data(), dupImages[5]->rowStride(),
        (unsigned char*)dupImages[6]->data(), dupImages[6]->rowStride(),
        (unsigned char*)dupImages[7]->data(), dupImages[7]->rowStride(),
        image.rows(), image.columns() );

    cudaCheckLastError( "Kirsch gradient computation kernel launch failed" );
    cudaMsgCheckError( cudaDeviceSynchronize(), "Kirsch gradient computation kernel run failed" );

    for( unsigned int i = 1; i < KERNEL_COUNT; ++i )
        delete dupImages[i];
}
