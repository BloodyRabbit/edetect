/** @file
 * @brief Template definition of class SeparableConvolutionFilter.
 *
 * @author Jan Bobek
 * @since 16th April 2015
 */

/*************************************************************************/
/* SeparableConvolutionFilter< RCF, CCF >                                */
/*************************************************************************/
template< typename RCF, typename CCF >
void
SeparableConvolutionFilter< RCF, CCF >::filter(
    IImage& image
    )
{
    mRowFilter.filter( image );
    mColumnFilter.filter( image );
}

template< typename RCF, typename CCF >
void
SeparableConvolutionFilter< RCF, CCF >::setParamVa(
    const char* name,
    va_list ap
    )
{
    static const char rpfx[] = { 'r', 'o', 'w', '-' };
    static const char cpfx[] = { 'c', 'o', 'l', 'u', 'm', 'n', '-' };

    if( !strncmp( name, rpfx, sizeof(rpfx) ) )
        mRowFilter.setParamVa( name + sizeof(rpfx), ap );
    else if( !strncmp( name, cpfx, sizeof(cpfx) ) )
        mColumnFilter.setParamVa( name + sizeof(cpfx), ap );
    else
    {
        va_list dup;
        va_copy( dup, ap );

        mRowFilter.setParamVa( name, ap );
        mColumnFilter.setParamVa( name, dup );

        va_end( dup );
    }
}

template< typename RCF, typename CCF >
void
SeparableConvolutionFilter< RCF, CCF >::setRowKernel(
    const float* kernel,
    unsigned int radius
    )
{
    mRowFilter.setKernel( kernel, radius );
}

template< typename RCF, typename CCF >
void
SeparableConvolutionFilter< RCF, CCF >::setColumnKernel(
    const float* kernel,
    unsigned int radius
    )
{
    mColumnFilter.setKernel( kernel, radius );
}
