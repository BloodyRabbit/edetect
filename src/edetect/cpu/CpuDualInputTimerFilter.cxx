/** @file
 * @brief Definition of class CpuDualInputTimerFilter.
 *
 * @author Jan Bobek
 * @since 6th May 2015
 */

#include "edetect.hxx"
#include "cpu/CpuDualInputTimerFilter.hxx"

/*************************************************************************/
/* CpuDualInputTimerFilter                                              */
/*************************************************************************/
CpuDualInputTimerFilter::CpuDualInputTimerFilter(
    IDualInputFilter* filter
    )
: IDualInputTimerFilter( filter )
{
}

void
CpuDualInputTimerFilter::filter2(
    IImage& dest,
    const IImage& src
    )
{
    float ms;
    timeval start, end;

    if( gettimeofday( &start, NULL ) )
        throw std::runtime_error(
            "CpuDualInputTimerFilter: starting call to `gettimeofday' failed" );

    IDualInputTimerFilter::filter2( dest, src );

    if( gettimeofday( &end, NULL ) )
        throw std::runtime_error(
            "CpuDualInputTimerFilter: ending call to `gettimeofday' failed" );

    timersub( &end, &start, &end );

    ms = end.tv_sec * 1e+3 +
        end.tv_usec * 1e-3;
    print( ms, dest );
}
