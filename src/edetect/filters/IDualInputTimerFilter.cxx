/** @file
 * @brief Definition of class IDualInputTimerFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "filters/IDualInputTimerFilter.hxx"

/*************************************************************************/
/* IDualInputTimerFilter                                                 */
/*************************************************************************/
IDualInputTimerFilter::IDualInputTimerFilter(
    IDualInputFilter* filter
    )
: mName( NULL ),
  mFilter( NULL )
{
    if( filter )
        setFilter( filter );
}

IDualInputTimerFilter::~IDualInputTimerFilter()
{
    free( mName );
    delete mFilter;
}

void
IDualInputTimerFilter::setParamVa(
    const char* name,
    va_list ap
    )
{
    const char* tname;
    IDualInputFilter* filter;

    if( !strcmp( name, "name" ) )
    {
        tname = va_arg( ap, const char* );
        setName( tname );
    }
    else if( !strcmp( name, "filter" ) )
    {
        filter = va_arg( ap, IDualInputFilter* );
        setFilter( filter );
    }
    else
        IDualInputFilter::setParamVa( name, ap );
}

void
IDualInputTimerFilter::setName(
    const char* name
    )
{
    free( mName );
    mName = strdup( name );
}

void
IDualInputTimerFilter::setFilter(
    IDualInputFilter* filter
    )
{
    delete mFilter;
    mFilter = filter;

    // Hijack the slave filters
    setFirst( mFilter->mFirst );
    mFilter->mFirst = NULL;
    setSecond( mFilter->mSecond );
    mFilter->mSecond = NULL;
}

void
IDualInputTimerFilter::filter2(
    IImage& dest,
    const IImage& src
    )
{
    mFilter->filter2( dest, src );
}

void
IDualInputTimerFilter::print(
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
