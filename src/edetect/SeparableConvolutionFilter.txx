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
    static const char rpfx[] = { 'r', 'o', 'w', '-' };
    static const char cpfx[] = { 'c', 'o', 'l', 'u', 'm', 'n', '-' };

    if( !strncmp( name, rpfx, sizeof(rpfx) ) )
        mRowFilter.setParam( name + sizeof(rpfx), value );
    else if( !strncmp( name, cpfx, sizeof(cpfx) ) )
        mColumnFilter.setParam( name + sizeof(cpfx), value );
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
