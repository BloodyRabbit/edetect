/** @file
 * @brief Template definition of class GaussianBlurFilter.
 *
 * @author Jan Bobek
 */

/*************************************************************************/
/* GaussianBlurFilter< F >                                               */
/*************************************************************************/
template< typename F >
GaussianBlurFilter< F >::GaussianBlurFilter()
: mKernel( NULL )
{
}

template< typename F >
GaussianBlurFilter< F >::~GaussianBlurFilter()
{
    delete mKernel;
}

template< typename F >
void
GaussianBlurFilter< F >::filter(
    IImage& image
    )
{
    mFilter.filter( image );
}

template< typename F >
void
GaussianBlurFilter< F >::setParam(
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
                "GaussianBlurFilter: Invalid radius value" );

        setRadius( radius );
    }
    else
        IImageFilter::setParam( name, value );
}

template< typename F >
void
GaussianBlurFilter< F >::setRadius(
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

    mFilter.setRowKernel( mKernel, radius );
    mFilter.setColumnKernel( mKernel, radius );
}
