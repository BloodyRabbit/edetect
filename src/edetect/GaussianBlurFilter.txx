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
    const unsigned int origin = radius;
    const unsigned int length = 2 * radius + 1;

    const double sigma = radius / 2.5;
    const double coef = -1.0 / (2.0 * sigma * sigma);

    delete mKernel;
    mKernel = new float[length];

    float sum = mKernel[origin] = 1.0f;
    for( unsigned int i = 1, r2i = 1; i <= radius; r2i += 1 + 2 * i++ )
        sum += 2.0f *
            (mKernel[origin - i] =
             mKernel[origin + i] =
             exp( coef * r2i ));

    for( unsigned int i = 0; i < length; ++i )
        mKernel[i] /= sum;

    mFilter.setRowKernel( mKernel, radius );
    mFilter.setColumnKernel( mKernel, radius );
}
