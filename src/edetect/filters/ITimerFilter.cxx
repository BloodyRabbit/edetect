/** @file
 * @brief Definition of class ITimerFilter.
 *
 * @author Jan Bobek
 * @since 1st May 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
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
ITimerFilter::filter(
    IImage& image
    )
{
    mFilter->filter( image );
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

void
ITimerFilter::print(
    float ms,
    const IImage& image
    )
{
    fputs( "Timer", stdout );
    if( mName )
        fprintf( stdout, " `%s'", mName );

    fprintf( stdout, ": %f milliseconds (%f MPix/s)\n",
             ms, 1e-3 * image.rows() * image.columns() / ms );
}
