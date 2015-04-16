/** @file
 * @brief Template definition of class SeparableConvolutionFilter.
 *
 * @author Jan Bobek
 * @since 16th April 2015
 */

/*************************************************************************/
/* SeparableConvolutionFilter< RF, CF >                                  */
/*************************************************************************/
template< typename RF, typename CF >
void
SeparableConvolutionFilter< RF, CF >::filter(
    IImage& image
    )
{
    mRowFilter.filter( image );
    mColumnFilter.filter( image );
}

template< typename RF, typename CF >
void
SeparableConvolutionFilter< RF, CF >::setParam(
    const char* name,
    const void* value
    )
{
    if( !strcmp( name, "row-kernel" ) )
        mRowFilter.setParam( "kernel", value );
    else if( !strcmp( name, "row-kernel-radius" ) )
        mRowFilter.setParam( "radius", value );
    else if( !strcmp( name, "column-kernel" ) )
        mColumnFilter.setParam( "kernel", value );
    else if( !strcmp( name, "column-kernel-radius" ) )
        mColumnFilter.setParam( "radius", value );
    else
        IImageFilter::setParam( name, value );
}

template< typename RF, typename CF >
void
SeparableConvolutionFilter< RF, CF >::setRowKernel(
    const float* kernel,
    unsigned int radius
    )
{
    mRowFilter.setKernel( kernel, radius );
}

template< typename RF, typename CF >
void
SeparableConvolutionFilter< RF, CF >::setColumnKernel(
    const float* kernel,
    unsigned int radius
    )
{
    mColumnFilter.setKernel( kernel, radius );
}
