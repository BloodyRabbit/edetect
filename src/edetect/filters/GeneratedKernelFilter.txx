/** @file
 * @brief Template definition of GeneratedKernelFilter.
 *
 * @author Jan Bobek
 * @since 16th April 2015
 */

/*************************************************************************/
/* GeneratedKernelFilter< K >                                            */
/*************************************************************************/
template< typename K >
GeneratedKernelFilter< K >::GeneratedKernelFilter(
    IImageFilter* filter,
    K kernel
    )
: mFilter( filter ),
  mKernel( kernel )
{
}

template< typename K >
GeneratedKernelFilter< K >::~GeneratedKernelFilter()
{
    delete mFilter;
}

template< typename K >
void
GeneratedKernelFilter< K >::filter(
    IImage& image
    )
{
    mFilter->filter( image );
}

template< typename K >
void
GeneratedKernelFilter< K >::setParamVa(
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

template< typename K >
void
GeneratedKernelFilter< K >::setRadius(
    unsigned int radius
    )
{
    unsigned int length;
    float* kernel = mKernel( radius, length );
    mFilter->setParam( "kernel", kernel, length );
}
