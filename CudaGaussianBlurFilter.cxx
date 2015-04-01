/** @file
 * @brief Definition of CudaGaussianBlurFilter class.
 *
 * @author Jan Bobek
 */

#include "common.hxx"
#include "CudaGaussianBlurFilter.hxx"

/*************************************************************************/
/* CudaGaussianBlurFilter                                                */
/*************************************************************************/
CudaGaussianBlurFilter::CudaGaussianBlurFilter(
    unsigned int radius
    )
{
    const double sigma = radius / 1.5;
    const double sigma2 = sigma * sigma;

    mKernel = new float[2 * radius + 1];

    float sum = mKernel[radius] = 1.0f;
    for( unsigned int i = 1; i <= radius; ++i )
        sum += 2.0f * (mKernel[radius - i] = mKernel[radius + i] = exp( -(double)(i * i) / sigma2 ));

    for( unsigned int i = 0; i < (2 * radius + 1); ++i )
        mKernel[i] /= sum;

    mFilter.setKernelRows( mKernel, radius );
    mFilter.setKernelColumns( mKernel, radius );
}

CudaGaussianBlurFilter::~CudaGaussianBlurFilter()
{
    delete mKernel;
}

void
CudaGaussianBlurFilter::process(
    CudaImage& image
    )
{
    mFilter.process( image );
}
