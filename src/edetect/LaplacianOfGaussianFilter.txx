/** @file
 * @brief Template definition of class LaplacianOfGaussianFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

/*************************************************************************/
/* LaplacianOfGaussianFilter< CF >                                       */
/*************************************************************************/
template< typename CF >
LaplacianOfGaussianFilter< CF >::LaplacianOfGaussianFilter()
: mKernel( NULL )
{
}

template< typename CF >
LaplacianOfGaussianFilter< CF >::~LaplacianOfGaussianFilter()
{
    free( mKernel );
}

template< typename CF >
void
LaplacianOfGaussianFilter< CF >::filter(
    IImage& image
    )
{
    mFilter.filter( image );
}

template< typename CF >
void
LaplacianOfGaussianFilter< CF >::setParam(
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
                "LaplacianOfGaussianFilter: Invalid radius value" );

        setRadius( radius );
    }
    else
        IImageFilter::setParam( name, value );
}

template< typename CF >
void
LaplacianOfGaussianFilter< CF >::setRadius(
    unsigned int radius
    )
{
    const unsigned int stride = 2 * radius + 1;
    const unsigned int origin = radius * (stride + 1);

    const double sigma = radius / 2.5;
    const double coef1 = 2.0 * sigma * sigma;
    const double coef2 = -1.0 / coef1;

    mKernel = (float*)realloc(
        mKernel, stride * stride * sizeof(*mKernel) );

    mKernel[origin] = -coef1;
    for( unsigned int i = 1, r2i = 1; i <= radius; r2i += 1 + 2 * i++ )
    {
        (mKernel[origin - i] =
         mKernel[origin - i * stride] =
         mKernel[origin + i] =
         mKernel[origin + i * stride] =
         (r2i - coef1) * exp( coef2 * r2i ));

        (mKernel[origin - i * (stride - 1)] =
         mKernel[origin - i * (stride + 1)] =
         mKernel[origin + i * (stride - 1)] =
         mKernel[origin + i * (stride + 1)] =
         (2 * r2i - coef1) * exp( coef2 * (2 * r2i) ));

        for( unsigned int j = i + 1, r2ij = r2i + j * j; j <= radius; r2ij += 1 + 2 * j++ )
            (mKernel[origin - i * stride - j] =
             mKernel[origin - i * stride + j] =
             mKernel[origin + i * stride - j] =
             mKernel[origin + i * stride + j] =

             mKernel[origin - j * stride - i] =
             mKernel[origin - j * stride + i] =
             mKernel[origin + j * stride - i] =
             mKernel[origin + j * stride + i] =

             (r2ij - coef1) * exp( coef2 * r2ij ));
    }

    mFilter.setKernel( mKernel, radius );
}
