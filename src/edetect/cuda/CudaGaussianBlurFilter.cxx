/** @file
 * @brief Definition of CudaGaussianBlurFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cuda/CudaGaussianBlurFilter.hxx"

/*************************************************************************/
/* CudaGaussianBlurFilter                                                */
/*************************************************************************/
CudaGaussianBlurFilter::CudaGaussianBlurFilter()
: mKernel( NULL )
{
}

CudaGaussianBlurFilter::~CudaGaussianBlurFilter()
{
    delete mKernel;
}

void
CudaGaussianBlurFilter::filter(
    CudaImage& image
    )
{
    mFilter.filter( image );
}

void
CudaGaussianBlurFilter::setParam(
    const char* name,
    const void* value
    )
{
    if( !strcmp( name, "radius" ) )
    {
        char* endptr;
        unsigned int radius =
            strtoul( (const char*)value, &endptr, 10 );

        if( *endptr )
            throw std::invalid_argument(
                "CudaGaussianBlurFilter: Invalid radius value" );

        setRadius( radius );
    }
    else
        IImageFilter::setParam( name, value );
}

void
CudaGaussianBlurFilter::setRadius(
    unsigned int radius
    )
{
    const double sigma = radius / 1.5;
    const double sigma2 = sigma * sigma;

    delete mKernel;
    mKernel = new float[2 * radius + 1];

    float sum = mKernel[radius] = 1.0f;
    for( unsigned int i = 1; i <= radius; ++i )
        sum += 2.0f * (mKernel[radius - i] = mKernel[radius + i] = exp( -(double)(i * i) / sigma2 ));

    for( unsigned int i = 0; i < (2 * radius + 1); ++i )
        mKernel[i] /= sum;

    mFilter.setParam( "row-kernel", mKernel );
    mFilter.setParam( "row-kernel-radius", &radius );
    mFilter.setParam( "column-kernel", mKernel );
    mFilter.setParam( "column-kernel-radius", &radius );
}
