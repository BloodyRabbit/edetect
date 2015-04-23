/** @file
 * @brief Definition of class IDualInputFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#include "edetect.hxx"
#include "IDualInputFilter.hxx"
#include "IImage.hxx"

/*************************************************************************/
/* IDualInputFilter                                                      */
/*************************************************************************/
IDualInputFilter::IDualInputFilter(
    IImageFilter* first,
    IImageFilter* second
    )
: mFirst( first ),
  mSecond( second )
{
}

IDualInputFilter::~IDualInputFilter()
{
    delete mFirst;
    delete mSecond;
}

void
IDualInputFilter::filter(
    IImage& image
    )
{
    IImage* dup = image.clone();

    if( mFirst )
        mFirst->filter( image );
    if( mSecond )
        mSecond->filter( *dup );

    filter2( image, *dup );
    delete dup;
}

void
IDualInputFilter::setParamVa(
    const char* name,
    va_list ap
    )
{
    IImageFilter* filter;

    if( !strcmp( name, "first" ) )
    {
        filter = va_arg( ap, IImageFilter* );
        setFirst( filter );
    }
    else if( !strcmp( name, "second" ) )
    {
        filter = va_arg( ap, IImageFilter* );
        setSecond( filter );
    }
    else
        IImageFilter::setParamVa( name, ap );
}

void
IDualInputFilter::setFirst(
    IImageFilter* filter
    )
{
    delete mFirst;
    mFirst = filter;
}

void
IDualInputFilter::setSecond(
    IImageFilter* filter
    )
{
    delete mSecond;
    mSecond = filter;
}
