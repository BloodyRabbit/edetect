/** @file
 * @brief Template definition of IGeneratedKernelFilter.
 *
 * @author Jan Bobek
 * @since 16th April 2015
 */

/*************************************************************************/
/* IGeneratedKernelFilter< CF >                                          */
/*************************************************************************/
template< typename CF >
IGeneratedKernelFilter< CF >::IGeneratedKernelFilter()
: mKernel( NULL )
{
}

template< typename CF >
IGeneratedKernelFilter< CF >::~IGeneratedKernelFilter()
{
    delete[] mKernel;
}

template< typename CF >
void
IGeneratedKernelFilter< CF >::filter(
    IImage& image
    )
{
    mFilter.filter( image );
}

template< typename CF >
void
IGeneratedKernelFilter< CF >::setParam(
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
                "IGeneratedKernelFilter: Invalid radius value" );

        setRadius( radius );
    }
    else
        IImageFilter::setParam( name, value );
}

template< typename CF >
void
IGeneratedKernelFilter< CF >::setRadius(
    unsigned int radius
    )
{
    delete[] mKernel;
    mKernel = generateKernel( radius );
    mFilter.setKernel( mKernel, radius );
}
