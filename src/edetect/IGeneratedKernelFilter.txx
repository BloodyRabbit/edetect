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
IGeneratedKernelFilter< CF >::setParamVa(
    const char* name,
    va_list ap
    )
{
    char* endptr;
    const char* strval;
    unsigned int radius;

    if( !strcmp( name, "radius" ) )
    {
        strval = va_arg( ap, const char* );
        radius = strtoul( strval, &endptr, 10 );

        if( *endptr )
            throw std::invalid_argument(
                "IGeneratedKernelFilter: Invalid radius value" );

        setRadius( radius );
    }
    else
        IImageFilter::setParamVa( name, ap );
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
