/** @file
 * @brief Definition of class ITimerFilter.
 *
 * @author Jan Bobek
 * @since 1st May 2015
 */

#include "edetect.hxx"
#include "filters/ITimerFilter.hxx"

/*************************************************************************/
/* ITimerFilter                                                          */
/*************************************************************************/
ITimerFilter::ITimerFilter(
    IImageFilter* filter
    )
: mName( NULL ),
  mFilter( filter )
{
}

ITimerFilter::~ITimerFilter()
{
    free( mName );
    delete mFilter;
}

void
ITimerFilter::setParamVa(
    const char* name,
    va_list ap
    )
{
    const char* tname;
    IImageFilter* filter;

    if( !strcmp( name, "name" ) )
    {
        tname = va_arg( ap, const char* );
        setName( tname );
    }
    else if( !strcmp( name, "filter" ) )
    {
        filter = va_arg( ap, IImageFilter* );
        setFilter( filter );
    }
    else
        IImageFilter::setParamVa( name, ap );
}

void
ITimerFilter::setName(
    const char* name
    )
{
    free( mName );
    mName = strdup( name );
}

void
ITimerFilter::setFilter(
    IImageFilter* filter
    )
{
    delete mFilter;
    mFilter = filter;
}
