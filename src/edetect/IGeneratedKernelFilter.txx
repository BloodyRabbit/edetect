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
    unsigned int radius;

    if( !strcmp( name, "radius" ) )
    {
        radius = va_arg( ap, unsigned int );
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
    unsigned int length;
    float* kernel = generateKernel( radius, length );
    mFilter.setKernel( kernel, length );
}
