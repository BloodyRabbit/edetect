/** @file
 * @brief Definition of class SeparableConvolutionFilter.
 *
 * @author Jan Bobek
 * @since 16th April 2015
 */

#include "edetect.hxx"
#include "filters/SeparableConvolutionFilter.hxx"

/*************************************************************************/
/* SeparableConvolutionFilter                                            */
/*************************************************************************/
SeparableConvolutionFilter::SeparableConvolutionFilter(
    IImageFilter* rowFilter,
    IImageFilter* columnFilter
    )
: mRowFilter( rowFilter ),
  mColumnFilter( columnFilter )
{
}

SeparableConvolutionFilter::~SeparableConvolutionFilter()
{
    delete mRowFilter;
    delete mColumnFilter;
}

void
SeparableConvolutionFilter::filter(
    IImage& image
    )
{
    mRowFilter->filter( image );
    mColumnFilter->filter( image );
}

void
SeparableConvolutionFilter::setParamVa(
    const char* name,
    va_list ap
    )
{
    static const char rpfx[] = { 'r', 'o', 'w', '-' };
    static const char cpfx[] = { 'c', 'o', 'l', 'u', 'm', 'n', '-' };

    if( !strncmp( name, rpfx, sizeof(rpfx) ) )
        mRowFilter->setParamVa( name + sizeof(rpfx), ap );
    else if( !strncmp( name, cpfx, sizeof(cpfx) ) )
        mColumnFilter->setParamVa( name + sizeof(cpfx), ap );
    else
    {
        va_list dup;
        va_copy( dup, ap );

        mRowFilter->setParamVa( name, ap );
        mColumnFilter->setParamVa( name, dup );

        va_end( dup );
    }
}
